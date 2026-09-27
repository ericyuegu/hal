"""Rotary transformer attention shared by 059 training and inference."""

import math
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
from torch.nn.attention.varlen import varlen_attn

AttnMask = Bool[Tensor, "B 1 L L"]


@dataclass(frozen=True, slots=True)
class TrunkConfig:
    """The trunk geometry. Experiment configs build one of these from their own fields."""

    d_model: int
    n_layers: int
    n_heads: int
    L_ctx: int
    attn_window: int = 0  # frames of look-back; 0 = full context
    # ``varlen_flash`` represents each row's ignored left prefix and real suffix as
    # separate causal sequences.  It therefore preserves the valid-token mask while
    # calling PyTorch's native FlashAttention kernel without a dense [B, L, L] mask.
    attention_backend: str = "varlen_flash"
    # Experiments that study depth parameterization can supply explicit
    # residual-branch multipliers.  ``None`` retains the historical attention
    # rule exactly; the MLP branch historically used 1.0.
    attention_scale: float | None = None
    mlp_scale: float = 1.0

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
        if self.attention_backend not in ("dense_sdpa", "varlen_flash"):
            raise ValueError(f"unknown attention_backend={self.attention_backend!r}")
        if self.attention_scale is not None and (not math.isfinite(self.attention_scale) or self.attention_scale <= 0):
            raise ValueError(f"attention_scale must be finite and positive, got {self.attention_scale!r}")
        if not math.isfinite(self.mlp_scale) or self.mlp_scale <= 0:
            raise ValueError(f"mlp_scale must be finite and positive, got {self.mlp_scale!r}")


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


def rotate_positions(x: Tensor, positions: Tensor, rotary: Rotary) -> Tensor:
    """Apply RoPE to absolute positions used by the bounded KV rings."""
    if positions.ndim not in (1, 2):
        raise ValueError("rotary positions must be [Q] or [B, Q]")
    frequencies = 1.0 / (
        rotary.base ** (torch.arange(0, rotary.dim, 2, device=x.device, dtype=torch.float32) / rotary.dim)
    )
    angles = positions.float()[..., None] * frequencies
    cosine = angles.cos().to(x.dtype)
    sine = angles.sin().to(x.dtype)
    if positions.ndim == 1:
        return apply_rotary_emb(x, cosine[None, :, None], sine[None, :, None])
    return apply_rotary_emb(x, cosine[:, :, None], sine[:, :, None])


@dataclass(frozen=True, slots=True)
class KVMemory:
    """Read view of a stream's bounded KV ring and its absolute positions."""

    kv: Tensor
    positions: Tensor
    query_position: Tensor
    window: int


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

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, L, _ = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_heads, self.head_dim)
        v = v.view(B, L, self.n_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin).transpose(1, 2)
        k = apply_rotary_emb(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def _project(self, y: Tensor, batch: int, length: int) -> Tensor:
        y = y.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.c_proj(y)

    def forward(
        self,
        x: Float[Tensor, "B L d_model"],
        mask: AttnMask | None,
        ctx_pad: Int[Tensor, " B"],
    ) -> Float[Tensor, "B L d_model"]:
        B, L, _ = x.shape
        q, k, v = self._qkv(x)
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
            y = cast(
                Tensor,
                varlen_attn(
                    q.transpose(1, 2).reshape(B * L, self.n_heads, self.head_dim),
                    k.transpose(1, 2).reshape(B * L, self.n_heads, self.head_dim),
                    v.transpose(1, 2).reshape(B * L, self.n_heads, self.head_dim),
                    cu_seqlens,
                    cu_seqlens,
                    L,
                    L,
                    window_size=window,
                ),
            ).reshape(B, L, self.n_heads, self.head_dim)
            return self.c_proj(y.reshape(B, L, self.d_model))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self._project(y, B, L)

    def forward_unpadded(self, x: Tensor) -> Tensor:
        """Use native causal attention when every context position is real."""
        B, L, _ = x.shape
        q, k, v = self._qkv(x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self._project(y, B, L)


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
        self.attn_scale = 1 / (2 * cfg.n_layers) ** 0.5 if cfg.attention_scale is None else cfg.attention_scale
        self.mlp_scale = cfg.mlp_scale

    def forward(
        self, x: Float[Tensor, "B L d_model"], mask: AttnMask | None, ctx_pad: Int[Tensor, " B"]
    ) -> Float[Tensor, "B L d_model"]:
        x = x + self.attn_scale * self.attn(rmsnorm(x), mask, ctx_pad)
        x = x + self.mlp_scale * self.mlp(rmsnorm(x))
        return x

    def forward_unpadded(self, x: Tensor) -> Tensor:
        x = x + self.attn_scale * self.attn.forward_unpadded(rmsnorm(x))
        return x + self.mlp_scale * self.mlp(rmsnorm(x))


class Trunk(nn.Module):
    """The 059 block stack with native varlen training and a dense reference."""

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.attn_window = cfg.attn_window
        self.L_ctx = cfg.L_ctx
        self.attention_backend = cfg.attention_backend
        self._attn_path: str | None = None

    @property
    def attn_path(self) -> str:
        return "unresolved" if self._attn_path is None else self._attn_path

    @torch.compiler.disable
    def resolve_attention(self, device_type: str) -> None:
        """Resolve the declared backend before compiling the complete model."""
        if self._attn_path is not None:
            return
        self._attn_path = (
            "varlen_flash" if self.attention_backend == "varlen_flash" and device_type == "cuda" else "dense"
        )
        logger.info(f"trunk attention: {self._attn_path} path, window={self.attn_window}")

    def _mask(self, ctx_pad: Int[Tensor, " B"], length: int) -> AttnMask | None:
        if self._attn_path is None:
            self.resolve_attention(ctx_pad.device.type)
        if self._attn_path == "varlen_flash":
            return None
        return dense_mask(ctx_pad, length, self.attn_window)

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
        """Run the dense SDPA correctness path."""
        self._check_shape(x, ctx_pad)
        return self._forward_with_mask(x, ctx_pad, dense_mask(ctx_pad, x.size(1), self.attn_window))

    def forward_unpadded(self, x: Tensor) -> Tensor:
        """Run the faster native causal path for batches without left padding."""
        if x.ndim != 3:
            raise ValueError(f"trunk takes x [B, L, d_model], got {tuple(x.shape)}")
        if self.attn_window:
            raise ValueError("the unpadded trunk path requires full causal attention")
        for block in self.blocks:
            x = cast(Block, block).forward_unpadded(x)
        return rmsnorm(x)

    def forward(self, x: Float[Tensor, "B L d_model"], ctx_pad: Int[Tensor, " B"]) -> Float[Tensor, "B L d_model"]:
        self._check_shape(x, ctx_pad)
        mask = self._mask(ctx_pad, x.size(1))
        return self._forward_with_mask(x, ctx_pad, mask)
