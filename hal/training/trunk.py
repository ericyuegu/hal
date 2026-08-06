"""The shared transformer trunk: rotary pre-norm blocks with causal attention.

Experiments 016, 019, 020 and 021 each carry a byte-identical copy of this stack, and 017 and 018
carry a variant with the same parameter creation order. Those files stay frozen, because
``hal.scripts.h2h`` rebuilds old checkpoints from them. New experiments import this module.

The trunk adds sliding-window attention (SWA) to that stack. ``TrunkConfig.attn_window`` gives the
number of frames a query can attend to, its own frame included, and ``0`` keeps the full context.
Two implementations make the same mask:

* The FlexAttention path builds a :class:`BlockMask` and runs a sparsity-aware kernel, so a window
  skips the masked blocks instead of computing and then discarding them.
* The dense path builds a ``[B, 1, L, L]`` bool mask for ``scaled_dot_product_attention``. It is the
  correctness reference in the tests, and the fallback on a box where FlexAttention cannot compile.

Both paths obey the same three rules: a key must not be in the future (``kv <= q``), must be inside
the window (``q - kv < attn_window``), and must not be in the sample's left-padded cold-start prefix
(``kv >= ctx_pad``). The diagonal stays open in all cases, because a fully masked query row makes
the attention output NaN.

Parameter creation order is the order in 016 to 021. A seeded build therefore draws the same
initial weights as those files do. ``tests/test_trunk.py`` pins this.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import jaxtyped
from loguru import logger
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention

# One compiled kernel per (batch, sequence) shape. The default limit is 8, and a run that passes it
# drops back to eager without a word: measured 4.7x slower and 2x the VRAM, while the reported path
# still says "flex".
torch._dynamo.config.cache_size_limit = 64

# Compilation is lazy, so both of these cost nothing until the first FlexAttention call. The mask
# build is compiled too: in eager it walks the whole [B, L, L] index grid once per forward, which
# measures 6.6 ms at B=64/L=1024 and 13.2 ms at B=32/L=2048, against 0.3 ms compiled.
_flex_attention = torch.compile(flex_attention, dynamic=False)
_create_block_mask = torch.compile(create_block_mask, dynamic=False)

AttnMask = Bool[Tensor, "B 1 L L"] | BlockMask


@dataclass(frozen=True, slots=True)
class TrunkConfig:
    """The trunk geometry. Experiment configs build one of these from their own fields."""

    d_model: int
    n_layers: int
    n_heads: int
    L_ctx: int
    attn_window: int = 0  # frames of look-back; 0 = full context
    # Fail instead of falling back to the dense path. A cloud run wants the fast kernel or an error,
    # not a quiet 4x slowdown; a dev box without it still wants to run.
    require_flex: bool = False

    def __post_init__(self) -> None:
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by positive n_heads={self.n_heads}")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError(f"rotary attention head_dim must be even, got {self.d_model // self.n_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be > 0, got {self.n_layers}")
        if self.L_ctx <= 0:
            raise ValueError(f"L_ctx must be > 0, got {self.L_ctx}")
        if self.attn_window < 0:
            raise ValueError(f"attn_window must be >= 0 (0 = full context), got {self.attn_window}")


class Rotary(nn.Module):
    inv_freq: Tensor
    cache_key: tuple[int, torch.device, torch.dtype] | None
    cos_cached: Tensor | None
    sin_cached: Tensor | None

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cache_key = None
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
        key = (seq_len, x.device, self.inv_freq.dtype)
        if key != self.cache_key:
            self.cache_key = key
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

    def at(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """RoPE factors for absolute positions used by incremental decoding."""
        freqs = torch.outer(positions.to(self.inv_freq), self.inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]


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


@jaxtyped(typechecker=beartype)
def dense_mask(ctx_pad: Int[Tensor, " B"], L: int, attn_window: int) -> Bool[Tensor, "B 1 L L"]:
    """The bool attention mask for ``scaled_dot_product_attention``: causal, inside the window, and
    clear of each sample's left-padded cold-start prefix. A padded query keeps its diagonal, so its
    row is never fully masked (SDPA would give NaN)."""
    idx = torch.arange(L, device=ctx_pad.device)
    keep = idx[:, None] >= idx[None, :]
    if attn_window > 0:
        keep = keep & (idx[:, None] - idx[None, :] < attn_window)
    key_real = idx[None, :] >= ctx_pad[:, None]
    diag = torch.eye(L, dtype=torch.bool, device=ctx_pad.device)
    return (keep[None] & (key_real[:, None, :] | diag[None]))[:, None]


def flex_mask_mod(ctx_pad: Int[Tensor, " B"], attn_window: int) -> Callable[..., Tensor]:
    """:func:`dense_mask`'s rule, written the way FlexAttention wants it: one predicate over index
    tensors. The tests compare it with the dense mask element by element."""

    def mask_mod(b: Tensor, h: Tensor, q: Tensor, kv: Tensor) -> Tensor:
        keep = (kv <= q) & (kv >= ctx_pad[b])
        if attn_window > 0:
            keep = keep & (q - kv < attn_window)
        return keep | (kv == q)

    return mask_mod


def block_mask(ctx_pad: Int[Tensor, " B"], L: int, attn_window: int) -> BlockMask:
    """The FlexAttention form of :func:`dense_mask`. FlexAttention drops whole masked blocks, so the
    cost of a window grows with the window and not with the sequence length."""
    mask_mod = flex_mask_mod(ctx_pad, attn_window)
    return _create_block_mask(mask_mod, ctx_pad.shape[0], None, L, L, device=ctx_pad.device)


@functools.cache
def flex_is_usable(device_type: str) -> bool:
    """Whether FlexAttention compiles on this box. Triton, the driver and the GPU all take part, so
    the probe is one small call, not a version comparison. The call includes a backward, because the
    forward alone runs on CPU but the backward does not."""
    try:
        q, k, v = (torch.zeros(1, 1, 128, 16, device=device_type, requires_grad=True) for _ in range(3))
        pad = torch.zeros(1, dtype=torch.long, device=device_type)
        _flex_attention(q, k, v, block_mask=block_mask(pad, 128, 0)).sum().backward()
    except (RuntimeError, NotImplementedError) as e:
        logger.warning(f"FlexAttention does not run on {device_type} ({type(e).__name__}: {e}); trunk uses SDPA")
        return False
    return True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: AttnMask) -> Float[Tensor, "B L d_model"]:
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if isinstance(mask, BlockMask):
            y = _flex_attention(q, k, v, block_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.c_proj(y)

    def forward_incremental(
        self, x: Tensor, past: tuple[Tensor, Tensor] | None, max_cache: int
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Causal attention over a rolling KV cache of the last ``max_cache`` frames."""
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        # Keep K unrotated in the cache and re-apply RoPE over the retained window with positions
        # 0..T-1. RoPE is relative, so the scores stay correct as the left edge slides.
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        if k.size(2) > max_cache:
            k = k[:, :, -max_cache:]
            v = v[:, :, -max_cache:]
        positions = torch.arange(k.size(2), device=x.device)
        cos, sin = self.rotary.at(positions)
        q = apply_rotary_emb(q, cos[:, -L:], sin[:, -L:])
        k_rot = apply_rotary_emb(k.transpose(1, 2), cos, sin).transpose(1, 2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k_rot, v)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.c_proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"]) -> Float[Tensor, "B L d_model"]:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_scale = 1 / (2 * cfg.n_layers) ** 0.5

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], mask: AttnMask) -> Float[Tensor, "B L d_model"]:
        x = x + self.attn_scale * self.attn(rmsnorm(x), mask)
        x = x + self.mlp(rmsnorm(x))
        return x

    def forward_incremental(
        self, x: Tensor, past: tuple[Tensor, Tensor] | None, max_cache: int
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn, kv = self.attn.forward_incremental(rmsnorm(x), past, max_cache)
        x = x + self.attn_scale * attn
        x = x + self.mlp(rmsnorm(x))
        return x, kv


class Trunk(nn.Module):
    """The block stack plus the final norm. Callers own the input projection and the heads.

    ``prefer_flex=False`` pins the dense reference path, which the tests compare against.
    """

    def __init__(self, cfg: TrunkConfig, *, prefer_flex: bool = True) -> None:
        super().__init__()
        if cfg.require_flex and not prefer_flex:
            raise ValueError("require_flex asks for the flex path and prefer_flex=False forbids it")
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.attn_window = cfg.attn_window
        # With a trained window the cache holds exactly the training window, so incremental decode
        # matches the full forward. Without one it holds the whole context.
        self.max_cache = cfg.attn_window if cfg.attn_window > 0 else cfg.L_ctx
        self.prefer_flex = prefer_flex
        self.require_flex = cfg.require_flex
        self._use_flex: bool | None = None

    @property
    def attn_path(self) -> str:
        """The attention path in use. The probe needs a device, so the answer is ``"unresolved"``
        until the first forward."""
        if self._use_flex is None:
            return "unresolved"
        return "flex" if self._use_flex else "dense"

    def _mask(self, ctx_pad: Int[Tensor, " B"], L: int) -> AttnMask:
        if self._use_flex is None:
            self._use_flex = self.prefer_flex and flex_is_usable(ctx_pad.device.type)
            if self.require_flex and not self._use_flex:
                raise RuntimeError(f"require_flex is set, but FlexAttention does not run on {ctx_pad.device.type}")
            logger.info(f"trunk attention: {'flex' if self._use_flex else 'dense'} path, window={self.attn_window}")
        if self._use_flex:
            return block_mask(ctx_pad, L, self.attn_window)
        return dense_mask(ctx_pad, L, self.attn_window)

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "B L d_model"], ctx_pad: Int[Tensor, " B"]) -> Float[Tensor, "B L d_model"]:
        mask = self._mask(ctx_pad, x.size(1))
        for block in self.blocks:
            x = block(x, mask)
        return rmsnorm(x)

    @torch.no_grad()
    def forward_incremental(
        self, x: Tensor, past: list[tuple[Tensor, Tensor] | None]
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Encode ONE new token against per-layer rolling KV state.

        The cached attention applies no mask, so every query sees the whole cache. That is correct
        for a single token and wrong for a chunk, where the tokens would also see each other. A
        chunk prefill needs its own mask; add it here when a caller needs one.

        Without a window, the cache holds ``L_ctx`` frames and the model was trained to read all of
        them, so a longer decode would drop history that the full forward keeps. That is a silent
        error of about 9e-2 in the hidden state, so it raises here instead."""
        if x.size(1) != 1:
            raise ValueError(f"incremental decode takes one token, got L={x.size(1)}")
        cached = 0 if past[0] is None else past[0][0].size(2)
        if self.attn_window == 0 and cached + 1 > self.max_cache:
            raise ValueError(
                f"incremental decode passed the {self.max_cache}-frame context at full attention: "
                "the cache would drop history the full forward keeps. Train with attn_window > 0."
            )
        new_past: list[tuple[Tensor, Tensor]] = []
        for block, old in zip(self.blocks, past, strict=True):
            x, kv = block.forward_incremental(x, old, self.max_cache)
            new_past.append(kv)
        return rmsnorm(x), new_past
