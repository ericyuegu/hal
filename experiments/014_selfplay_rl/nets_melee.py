"""Melee policy/value network, warm-started byte-exactly from the 012 IL checkpoint.

The trunk (rotary causal GPT over per-frame tokens) is COPIED verbatim from
``experiments/012_multi_token.py`` — experiments can't import each other, and
checkpoint compatibility is the contract, so the math here must not drift from
012's. ``PolicyValueNet`` wraps that trunk with an RL-shaped output surface:

* ``policy_head`` — the deployed offset-1 head lifted out of 012's multi-token
  ``heads`` stack (the far-horizon auxiliary heads are dropped; they were a
  training-only signal). Its weights load VERBATIM from ``heads.{primary_idx}``.
* ``value_head`` — a fresh zero-initialised scalar critic. Zero init means the
  warm-started policy predicts identical logits to 012 while V(s) starts at 0
  and is learned during the value-warmup phase.

``load_il_policy`` remaps a 012 checkpoint onto this module and loads it with
``strict=True`` (value-head zeros are ADDED to the remapped dict rather than
tolerated via ``strict=False``) so any key/shape drift fails loud.

``FactoredCategorical`` is the single action-distribution definition shared by
the rollout collector and the PPO learner: a product of the four independent
group categoricals (buttons / main-stick / c-stick / triggers) that 012's
355-logit vector concatenates.

Tensor-dim names: B batch, L / L_ctx sequence, d_model, n_heads, d_head,
half_dim (= d_head/2), n_groups (= 4), A_VOCAB (= 355).
"""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from typing import runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from torch import Tensor

from hal.training import scoring
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import CAT_FEATURES
from hal.training.features import FLOAT_FEATURES
from hal.training.features import Context

# Action-vector channel split (A_DIM=14): [0:6] sticks+triggers, [6:14] buttons {0,1}.
_N_CONT = 6
_N_BUTTONS = A_DIM - _N_CONT

# Per-frame token: all four players' gamestate concatenated in the feature dim.
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


# %%
# --- action-group discretizer (copied verbatim from 012) ---------------------
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
# --- GPT backbone (copied verbatim from 012: rotary, RMSNorm, causal SDPA) ----
@dataclass(frozen=True, slots=True)
class ArchConfig:
    """The trunk-identity knobs — everything ``PolicyValueNet`` needs to rebuild
    012's backbone shape. Built from a 012 checkpoint's saved ``cfg`` dict via
    ``from_012_cfg`` (which reads ONLY these model-identity fields, ignoring the
    checkpoint's optimization/eval/host knobs)."""

    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    L_ctx: int = 256
    char_vocab: int = 32
    char_dim: int = 12
    stage_vocab: int = 32
    stage_dim: int = 4

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")

    @staticmethod
    def from_012_cfg(cfg: dict) -> ArchConfig:
        """Reconstruct the trunk shape from a 012 checkpoint's ``cfg`` dict. Only the
        model-identity fields are read; optimization/eval/host keys are ignored so a
        checkpoint saved by any 012 code version reconstructs the same architecture."""
        fields = ("d_model", "n_layers", "n_heads", "L_ctx", "char_vocab", "char_dim", "stage_vocab", "stage_dim")
        missing = [k for k in fields if k not in cfg]
        if missing:
            raise KeyError(f"012 cfg missing trunk-identity fields: {missing}")
        return ArchConfig(**{k: cfg[k] for k in fields})

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


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
def apply_rotary_at(
    x: Float[Tensor, "n L n_heads head_dim"],
    cos: Float[Tensor, "n L 1 half_dim"],
    sin: Float[Tensor, "n L 1 half_dim"],
) -> Float[Tensor, "n L n_heads head_dim"]:
    """Rotary at explicit per-token absolute positions (KV-cache incremental path).

    012's ``Rotary`` caches cos/sin against a contiguous ``arange(seq_len)``; here
    each slot's new token sits at its own absolute position, so ``cos``/``sin`` are
    gathered per-slot from a precomputed table (see ``SlotCaches`` rope table)
    rather than shared across a batch. Same half-split rotation as ``apply_rotary_emb``,
    so a token rotated here at position ``p`` matches a full forward that placed it at
    ``arange`` index ``p`` — the shift-invariance the exact-growth test pins."""
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


@jaxtyped(typechecker=beartype)
def rmsnorm(x0: Float[Tensor, "... d"], eps: float = 1e-6) -> Float[Tensor, "... d"]:
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    @jaxtyped(typechecker=beartype)
    def qkv(
        self, x: Float[Tensor, "n L d_model"]
    ) -> tuple[
        Float[Tensor, "n L n_heads head_dim"],
        Float[Tensor, "n L n_heads head_dim"],
        Float[Tensor, "n L n_heads head_dim"],
    ]:
        """Fused projection + head-split, WITHOUT rotary (callers rotate: the full
        path via ``self.rotary``, the incremental path via ``apply_rotary_at``)."""
        n, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(n, L, self.n_heads, self.head_dim)
        k = k.view(n, L, self.n_heads, self.head_dim)
        v = v.view(n, L, self.n_heads, self.head_dim)
        return q, k, v

    @jaxtyped(typechecker=beartype)
    def attend(
        self,
        q: Float[Tensor, "n Lq n_heads head_dim"],
        k: Float[Tensor, "n Lk n_heads head_dim"],
        v: Float[Tensor, "n Lk n_heads head_dim"],
        mask: Bool[Tensor, "n 1 Lq Lk"],
    ) -> Float[Tensor, "n Lq d_model"]:
        """SDPA over already-rotated q/k + v, then the output projection. Shared by the
        full path (Lq==Lk, causal mask) and the incremental path (Lq==1 against the
        cache, key-padding mask)."""
        n, Lq, _, _ = q.shape
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(n, Lq, self.d_model)
        return self.c_proj(y)

    @jaxtyped(typechecker=beartype)
    def forward_capture(
        self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]
    ) -> tuple[
        Float[Tensor, "B L d_model"],
        Float[Tensor, "B L n_heads head_dim"],
        Float[Tensor, "B L n_heads head_dim"],
    ]:
        """Full forward that ALSO returns the post-RoPE k, v so ``rebuild`` can seed the
        cache from a batched forward. Identical math to ``forward``."""
        q, k, v = self.qkv(x)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = self.attend(q, k, v, mask)
        return y, k, v

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]) -> Float[Tensor, "B L d_model"]:
        y, _, _ = self.forward_capture(x, mask)
        return y


class MLP(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "n L d_model"]) -> Float[Tensor, "n L d_model"]:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_scale = 1 / (2 * cfg.n_layers) ** 0.5

    @jaxtyped(typechecker=beartype)
    def forward_capture(
        self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]
    ) -> tuple[
        Float[Tensor, "B L d_model"],
        Float[Tensor, "B L n_heads head_dim"],
        Float[Tensor, "B L n_heads head_dim"],
    ]:
        y, k, v = self.attn.forward_capture(rmsnorm(x), mask)
        x = x + self.attn_scale * y
        x = x + self.mlp(rmsnorm(x))
        return x, k, v

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: Bool[Tensor, "B 1 L L"]) -> Float[Tensor, "B L d_model"]:
        x, _, _ = self.forward_capture(x, mask)
        return x


# %%
@runtime_checkable
class Critic(Protocol):
    """Value-estimation seam. The default reads the ego trunk hidden; a centralized
    (MAPPO) critic that sees both ports slots in behind the same call without
    touching the actor path."""

    def values(self, hidden: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch"]: ...


class PolicyValueNet(nn.Module):
    """012's trunk + an RL output surface (offset-1 policy head + a scalar value head).

    The backbone submodule names/shapes match 012's ``GPT`` exactly so a 012 state
    dict loads verbatim (see ``load_il_policy``). ``forward_full`` reproduces 012's
    per-frame backbone hidden; ``kv_cache`` reuses the same blocks for incremental
    decode."""

    def __init__(self, cfg: ArchConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.L_ctx = cfg.L_ctx

        self.cat_embeds = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in CAT_FEATURES.items()}
        )
        self.char_emb = nn.Embedding(cfg.char_vocab, cfg.char_dim)
        self.stage_emb = nn.Embedding(cfg.stage_vocab, cfg.stage_dim)
        per_player = len(FLOAT_FEATURES) * 2 + sum(dim for _, dim in CAT_FEATURES.values())  # float+mask+cat
        d_in = len(_PLAYER_PREFIXES) * per_player + A_DIM + 2 * cfg.char_dim + cfg.stage_dim

        self.ctx_proj = nn.Linear(d_in, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        self.policy_head = nn.Linear(cfg.d_model, A_VOCAB)
        self.value_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trig_centers", scoring.TRIGGER_CENTERS.clone())

    # --- feature assembly (copied verbatim from 012's GPT) -------------------
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

    def context_tokens(self, features: dict[str, Tensor]) -> Float[Tensor, "B L d_model"]:
        parts = [self._per_player_features(features, p) for p in _PLAYER_PREFIXES]
        parts.append(torch.cat([features[f"ego_{ch}"][..., None] for ch in ACTION_CHANNELS], dim=-1))
        parts.append(self.char_emb(features["ego_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.char_emb(features["opp_character"].clamp(0, self.char_emb.num_embeddings - 1)))
        parts.append(self.stage_emb(features["stage"].clamp(0, self.stage_emb.num_embeddings - 1)))
        return self.ctx_proj(torch.cat(parts, dim=-1))

    def attn_mask(self, ctx_pad: Int[Tensor, " B"], L: int, device: torch.device) -> Bool[Tensor, "B 1 L L"]:
        """Causal mask that also hides each sample's left-padded cold-start prefix (key < ctx_pad).
        A padded query keeps its diagonal so its row is never fully masked (SDPA would NaN)."""
        idx = torch.arange(L, device=device)
        causal = idx[:, None] >= idx[None, :]
        key_real = idx[None, :] >= ctx_pad[:, None]
        diag = torch.eye(L, dtype=torch.bool, device=device)
        return (causal[None] & (key_real[:, None, :] | diag[None]))[:, None]

    @jaxtyped(typechecker=beartype)
    def forward_full(self, ctx: Context) -> Float[Tensor, "B L d_model"]:
        """012's backbone hidden: one rmsnorm'd vector per frame. Heads applied by callers."""
        x = self.context_tokens(ctx.features)
        mask = self.attn_mask(ctx.ctx_pad, x.size(1), x.device)
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)

    @jaxtyped(typechecker=beartype)
    def forward_full_capture(self, ctx: Context) -> tuple[Float[Tensor, "B L d_model"], list[tuple[Tensor, Tensor]]]:
        """``forward_full`` that also returns each block's post-RoPE (k, v), so
        ``kv_cache.rebuild`` can batch-seed the caches from one full forward."""
        x = self.context_tokens(ctx.features)
        mask = self.attn_mask(ctx.ctx_pad, x.size(1), x.device)
        kvs: list[tuple[Tensor, Tensor]] = []
        for block in self.blocks:
            x, k, v = block.forward_capture(x, mask)
            kvs.append((k, v))
        return rmsnorm(x), kvs

    @jaxtyped(typechecker=beartype)
    def policy_logits(self, hidden: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch A_vocab"]:
        return self.policy_head(hidden).float()

    @jaxtyped(typechecker=beartype)
    def values(self, hidden: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch"]:
        return self.value_head(hidden).squeeze(-1).float()


class SharedTrunkCritic:
    """Default ``Critic``: V(s) from the ego trunk hidden via the net's value head."""

    def __init__(self, net: PolicyValueNet) -> None:
        self.net = net

    @jaxtyped(typechecker=beartype)
    def values(self, hidden: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch"]:
        return self.net.values(hidden)


# %%
class FactoredCategorical:
    """Product of the four independent group categoricals over 012's 355-logit vector.

    Slices the concatenated logits at the fixed group offsets and treats each group
    (buttons / main-stick / c-stick / triggers) as its own categorical, conditionally
    independent given context. Log-probs, entropies and KLs sum across groups. One
    definition, used by both the rollout collector and the PPO learner."""

    @jaxtyped(typechecker=beartype)
    def __init__(self, logits: Float[Tensor, "*batch A_vocab"]) -> None:
        if logits.shape[-1] != A_VOCAB:
            raise ValueError(f"expected {A_VOCAB} logits, got {logits.shape[-1]}")
        self.logits = logits
        # Per-group log-softmax over each slice; kept as a list of [*batch, vocab_g].
        self._log_probs: list[Tensor] = [
            F.log_softmax(logits[..., off : off + vocab], dim=-1)
            for off, vocab in zip(_GROUP_OFFSETS, _GROUP_VOCABS, strict=True)
        ]

    @jaxtyped(typechecker=beartype)
    def log_prob(self, idx: Int[Tensor, "*batch n_groups"]) -> Float[Tensor, "*batch"]:
        """Joint log-prob = sum of the per-group class log-probs."""
        total = self._log_probs[0].new_zeros(idx.shape[:-1])
        for g, lp in enumerate(self._log_probs):
            total = total + lp.gather(-1, idx[..., g : g + 1].long()).squeeze(-1)
        return total

    @jaxtyped(typechecker=beartype)
    def entropy(self) -> Float[Tensor, "*batch"]:
        """Joint entropy = sum of the per-group entropies (groups independent)."""
        total: Tensor | None = None
        for lp in self._log_probs:
            h = -(lp.exp() * lp).sum(-1)
            total = h if total is None else total + h
        assert total is not None
        return total

    @jaxtyped(typechecker=beartype)
    def sample(self, *, generator: torch.Generator | None = None) -> Int[Tensor, "*batch n_groups"]:
        """One class per group, sampled independently → ``[*batch, 4]`` indices."""
        picks: list[Tensor] = []
        for lp in self._log_probs:
            probs = lp.exp().reshape(-1, lp.shape[-1])
            draw = torch.multinomial(probs, 1, generator=generator).reshape(lp.shape[:-1])
            picks.append(draw)
        return torch.stack(picks, dim=-1)

    @jaxtyped(typechecker=beartype)
    def kl_to(self, other: "FactoredCategorical") -> Float[Tensor, "*batch"]:  # noqa: UP037
        """KL(self || other) = sum of per-group KLs. The KL-to-IL penalty term for M4."""
        total: Tensor | None = None
        for lp, lq in zip(self._log_probs, other._log_probs, strict=True):
            kl = (lp.exp() * (lp - lq)).sum(-1)
            total = kl if total is None else total + kl
        assert total is not None
        return total


# %%
def load_il_policy(ckpt_path: Path) -> tuple[PolicyValueNet, ArchConfig]:
    """Warm-start a ``PolicyValueNet`` from a 012 checkpoint, byte-exactly.

    The backbone (embeddings, ctx_proj, blocks incl. ``attn.rotary.inv_freq``, and
    the stick/trigger center buffers) loads verbatim; 012's offset-1 head
    (``heads.{index of 1}``) becomes ``policy_head``; the far-horizon auxiliary heads
    are dropped; a fresh zero value head is added. The remapped dict is loaded with
    ``strict=True`` so any drift fails loud rather than silently zero-initialising."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ArchConfig.from_012_cfg(state["cfg"])
    offsets = tuple(state["cfg"]["head_offsets"])
    if 1 not in offsets:
        raise ValueError(f"012 checkpoint has no offset-1 head to deploy: {offsets}")
    primary = offsets.index(1)

    src = state["model"]
    remapped: dict[str, Tensor] = {}
    head_prefix = f"heads.{primary}."
    for k, v in src.items():
        if k.startswith("heads."):
            if k.startswith(head_prefix):
                remapped["policy_head." + k[len(head_prefix) :]] = v
            # else: auxiliary head — dropped.
        else:
            remapped[k] = v  # backbone verbatim (incl. rotary.inv_freq + center buffers)

    net = PolicyValueNet(cfg)
    # Add the fresh zero value head so strict=True sees an exact key/shape match.
    remapped["value_head.weight"] = net.value_head.weight.detach().clone()
    remapped["value_head.bias"] = net.value_head.bias.detach().clone()
    net.load_state_dict(remapped, strict=True)
    net.eval()
    return net, cfg
