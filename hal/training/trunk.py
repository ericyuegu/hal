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

Shapes are annotated but not checked at runtime. Dynamo cannot trace the runtime type wrapper.
The public forward method has one direct shape guard for the error that could otherwise broadcast.
``tests/test_trunk.py`` checks all attention paths against a reference.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool
from jaxtyping import Float
from jaxtyping import Int
from loguru import logger
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.varlen import varlen_attn

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
    # ``varlen_flash`` represents each row's ignored left prefix and real suffix as
    # separate causal sequences.  It therefore preserves the valid-token mask while
    # calling PyTorch's native FlashAttention kernel without a dense [B, L, L] mask.
    attention_backend: str = "auto_flex"

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
        if self.attention_backend not in ("auto_flex", "dense_sdpa", "varlen_flash"):
            raise ValueError(f"unknown attention_backend={self.attention_backend!r}")
        if self.require_flex and self.attention_backend != "auto_flex":
            raise ValueError("require_flex is compatible only with attention_backend='auto_flex'")


class Rotary(nn.Module):
    inv_freq: Tensor
    cache_key: tuple[int, torch.device, torch.dtype] | None
    cos_cached: Tensor | None
    sin_cached: Tensor | None

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cache_key = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(
        self, x: Float[Tensor, "B L n_heads head_dim"]
    ) -> tuple[
        Float[Tensor, "1 L 1 half_dim"],
        Float[Tensor, "1 L 1 half_dim"],
    ]:
        return self.at(x.shape[1], x.device, x.dtype)

    def at(self, length: int, device: torch.device, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
        """RoPE factors for absolute positions ``0..length-1``, in the module's dtype.

        The angles are ALWAYS built at fp32, from the integer geometry rather than from the
        ``inv_freq`` buffer, because this table is a lookup and not a weight. A whole-module cast to
        fp16 (what an eval decode cast leaves behind) puts 4e-4 of relative slack on a frequency,
        which the position multiplies into a phase error of 2e-2 over a 128-frame window and 3.9e-1
        over 1024 — against the 2.4e-4 that rounding the finished table costs. An fp16 position
        counter also stops being exact past 2048 frames. The buffer stays registered, and stays the
        source whenever it is still fp32, so neither the checkpoint keys nor the fp32 arithmetic
        move."""
        output_dtype = self.inv_freq.dtype if dtype is None else dtype
        key = (length, device, output_dtype)
        if torch.compiler.is_compiling():
            # A cached tensor created while CUDA Graph capture is tracing becomes
            # graph-owned storage. Saving it on the module makes the next replay
            # read storage that the graph has already overwritten. Compiled paths
            # keep the same RoPE arithmetic but own the factors inside the graph.
            inv_freq = self.inv_freq
            if inv_freq.dtype != torch.float32:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
            return (
                freqs.cos().to(output_dtype)[None, :, None, :],
                freqs.sin().to(output_dtype)[None, :, None, :],
            )
        if key != self.cache_key:
            inv_freq = self.inv_freq
            if inv_freq.dtype != torch.float32:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            freqs = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inv_freq)
            self.cache_key = key
            self.cos_cached = freqs.cos().to(output_dtype)[None, :, None, :]
            self.sin_cached = freqs.sin().to(output_dtype)[None, :, None, :]
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached, self.sin_cached


def apply_rotary_emb(
    x: Float[Tensor, "B L n_heads head_dim"],
    cos: Float[Tensor, "1 L 1 half_dim"],
    sin: Float[Tensor, "1 L 1 half_dim"],
) -> Float[Tensor, "B L n_heads head_dim"]:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


def rmsnorm(x0: Float[Tensor, "... d"], eps: float = 1e-6) -> Float[Tensor, "... d"]:
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)


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
    forward alone runs on CPU but the backward does not.

    ``enable_grad`` because the first caller is often an eval worker, whose first forward runs under
    ``torch.no_grad()``. Without it the backward raises there, the probe reads that as "no flex", and
    the answer is cached for the life of the process."""
    try:
        with torch.enable_grad():
            q, k, v = (torch.zeros(1, 1, 128, 16, device=device_type, requires_grad=True) for _ in range(3))
            pad = torch.zeros(1, dtype=torch.long, device=device_type)
            cast(Tensor, _flex_attention(q, k, v, block_mask=block_mask(pad, 128, 0))).sum().backward()
    except (RuntimeError, NotImplementedError) as e:
        logger.warning(f"FlexAttention does not run on {device_type} ({type(e).__name__}: {e}); trunk uses SDPA")
        return False
    return True


@functools.cache
def varlen_flash_is_usable(device_type: str, L: int, n_heads: int, head_dim: int) -> bool:
    """Probe the exact native FlashAttention forward/backward geometry once."""
    if device_type != "cuda":
        return False
    try:
        q = torch.zeros(L * 2, n_heads, head_dim, device=device_type, dtype=torch.bfloat16, requires_grad=True)
        k = torch.zeros_like(q, requires_grad=True)
        v = torch.zeros_like(q, requires_grad=True)
        # Includes both a zero-length prefix and a one-token real suffix.
        lengths = torch.tensor((0, L, L - 1, 1), device=device_type, dtype=torch.int32)
        cumulative = lengths.cumsum(0, dtype=torch.int32)
        cu_seqlens = torch.cat((cumulative.new_zeros(1), cumulative))
        cast(Tensor, varlen_attn(q, k, v, cu_seqlens, cu_seqlens, L, L, window_size=(-1, 0))).sum().backward()
        torch.cuda.synchronize()
    except (RuntimeError, NotImplementedError) as exc:
        logger.warning(f"native varlen FlashAttention probe failed ({type(exc).__name__}: {exc})")
        return False
    return True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads
        self.attn_window = cfg.attn_window
        self.attention_backend = cfg.attention_backend
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(
        self,
        x: Float[Tensor, "B L d_model"],
        mask: AttnMask | None,
        ctx_pad: Int[Tensor, " B"],
    ) -> Float[Tensor, "B L d_model"]:
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if self.attention_backend == "varlen_flash" and x.device.type == "cuda":
            if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
                raise RuntimeError(
                    "varlen_flash requires matching FP16/BF16 QKV activations; "
                    f"got q={q.dtype}, k={k.dtype}, v={v.dtype}. "
                    "Run the model forward inside its configured AMP context."
                )
            # Keep a static B*L token shape.  Each ignored prefix and real suffix is
            # an independent causal sequence, so real queries cannot see padded keys.
            # Zero-length prefix sequences are valid and keep cu_seqlens fixed at 2B+1.
            lengths = torch.stack((ctx_pad, L - ctx_pad), dim=1).reshape(-1).to(torch.int32)
            cumulative = lengths.cumsum(0, dtype=torch.int32)
            cu_seqlens = torch.cat((cumulative.new_zeros(1), cumulative))
            window = (-1, 0) if self.attn_window == 0 else (self.attn_window - 1, 0)
            y = varlen_attn(
                q.transpose(1, 2).reshape(B * L, self.n_heads, self.head_dim),
                k.transpose(1, 2).reshape(B * L, self.n_heads, self.head_dim),
                v.transpose(1, 2).reshape(B * L, self.n_heads, self.head_dim),
                cu_seqlens,
                cu_seqlens,
                L,
                L,
                window_size=window,
            ).reshape(B, L, self.n_heads, self.head_dim)
            return self.c_proj(y.reshape(B, L, self.d_model))
        if isinstance(mask, BlockMask):
            y = cast(Tensor, _flex_attention(q, k, v, block_mask=mask))
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: Float[Tensor, "B L d_model"]) -> Float[Tensor, "B L d_model"]:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
        self.attn_scale = 1 / (2 * cfg.n_layers) ** 0.5

    def forward(
        self, x: Float[Tensor, "B L d_model"], mask: AttnMask | None, ctx_pad: Int[Tensor, " B"]
    ) -> Float[Tensor, "B L d_model"]:
        x = x + self.attn_scale * self.attn(rmsnorm(x), mask, ctx_pad)
        x = x + self.mlp(rmsnorm(x))
        return x


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
        self.L_ctx = cfg.L_ctx
        self.prefer_flex = prefer_flex
        self.require_flex = cfg.require_flex
        self.attention_backend = cfg.attention_backend
        self._use_flex: bool | None = None

    @property
    def attn_path(self) -> str:
        """The attention path in use. The probe needs a device, so the answer is ``"unresolved"``
        until the first forward."""
        if self._use_flex is None:
            return "unresolved"
        if self.attention_backend == "varlen_flash":
            return "varlen_flash"
        return "flex" if self._use_flex else "dense"

    @torch.compiler.disable
    def _resolve_attn_path(self, device_type: str) -> None:
        """Run the one-time hardware probe and path announcement outside Dynamo tracing.

        The probe is cached with :func:`functools.cache`, whose wrapper Dynamo deliberately ignores,
        and Loguru finds its caller with :func:`sys._getframe`, which Dynamo cannot trace. A trunk's
        first forward is commonly already inside ``torch.compile``, so keep both eager explicitly.
        Later forwards skip this method once ``_use_flex`` has been resolved.
        """
        self._use_flex = self.attention_backend == "auto_flex" and self.prefer_flex and flex_is_usable(device_type)
        if self.require_flex and not self._use_flex:
            raise RuntimeError(f"require_flex is set, but FlexAttention does not run on {device_type}")
        path = (
            "varlen_flash"
            if self.attention_backend == "varlen_flash" and device_type == "cuda"
            else ("flex" if self._use_flex else "dense")
        )
        logger.info(f"trunk attention: {path} path, window={self.attn_window}")

    def _mask(self, ctx_pad: Int[Tensor, " B"], L: int) -> AttnMask | None:
        if self._use_flex is None:
            self._resolve_attn_path(ctx_pad.device.type)
        if self.attention_backend == "varlen_flash" and ctx_pad.device.type == "cuda":
            return None
        if self._use_flex:
            return block_mask(ctx_pad, L, self.attn_window)
        return dense_mask(ctx_pad, L, self.attn_window)

    @staticmethod
    def _check_shape(x: Tensor, ctx_pad: Tensor) -> None:
        # The one runtime shape check on the trunk's input path, at the per-STEP boundary. A ctx_pad
        # of the wrong length does not raise further down, it BROADCASTS — one sample's cold-start
        # prefix silently masks every sample. 0.3 us, and dynamo traces it, which a jaxtyped wrapper
        # is not (it raises on the traced tensor, so a checked forward cannot be compiled).
        if x.ndim != 3 or ctx_pad.shape != x.shape[:1]:
            raise ValueError(
                f"trunk takes x [B, L, d_model] and ctx_pad [B]; got {tuple(x.shape)}, {tuple(ctx_pad.shape)}"
            )

    def _forward_with_mask(
        self,
        x: Float[Tensor, "B L d_model"],
        ctx_pad: Int[Tensor, " B"],
        mask: AttnMask | None,
    ) -> Float[Tensor, "B L d_model"]:
        for block in self.blocks:
            x = block(x, mask, ctx_pad)
        return rmsnorm(x)

    def forward_dense(
        self, x: Float[Tensor, "B L d_model"], ctx_pad: Int[Tensor, " B"]
    ) -> Float[Tensor, "B L d_model"]:
        """Run the dense-SDPA correctness path without resolving FlexAttention."""
        self._check_shape(x, ctx_pad)
        return self._forward_with_mask(x, ctx_pad, dense_mask(ctx_pad, x.size(1), self.attn_window))

    def forward(self, x: Float[Tensor, "B L d_model"], ctx_pad: Int[Tensor, " B"]) -> Float[Tensor, "B L d_model"]:
        self._check_shape(x, ctx_pad)
        mask = self._mask(ctx_pad, x.size(1))
        return self._forward_with_mask(x, ctx_pad, mask)
