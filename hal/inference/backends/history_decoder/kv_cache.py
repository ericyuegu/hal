"""Bounded attention KV rings with persistent positions across window eviction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from hal.inference.cuda_graph import CapturedCall
from hal.training.trunk import Block
from hal.training.trunk import Rotary
from hal.training.trunk import apply_rotary_emb
from hal.training.trunk import rmsnorm

if TYPE_CHECKING:
    from hal.inference.backends.history_decoder.model import GPT


@torch.library.custom_op("hal::append_kv", mutates_args=("cache",))
def append_kv(cache: Tensor, slots: Tensor, values: Tensor) -> None:
    """Keep functionalization from turning a small update into a cache-sized copy."""
    cache.index_copy_(3, slots, values)


@append_kv.register_fake
def _append_kv_fake(cache: Tensor, slots: Tensor, values: Tensor) -> None:
    pass


def rotate_positions(x: Tensor, positions: Tensor, rotary: Rotary) -> Tensor:
    frequencies = 1.0 / (
        rotary.base ** (torch.arange(0, rotary.dim, 2, device=x.device, dtype=torch.float32) / rotary.dim)
    )
    angles = positions.float()[:, None] * frequencies[None, :]
    return apply_rotary_emb(x, angles.cos().to(x.dtype)[None, :, None], angles.sin().to(x.dtype)[None, :, None])


@dataclass(frozen=True, slots=True)
class KVMemory:
    kv: Tensor
    positions: Tensor
    query_position: Tensor
    window: int


class KVCache:
    def __init__(self, model: GPT, update_frames: int, device: torch.device) -> None:
        if update_frames not in (1, 2):
            raise ValueError("KV cache updates must contain one or two frames")
        arch = model.cfg.arch
        self.window = arch.L_ctx
        # The first query in a two-token update still needs the oldest extra key.
        self.capacity = self.window + update_frames - 1
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.layers = tuple(
            torch.zeros(2, 1, arch.n_heads, self.capacity, arch.d_model // arch.n_heads, device=device, dtype=dtype)
            for _ in range(arch.n_layers)
        )
        self.history = torch.zeros(
            2,
            1,
            arch.temporal_heads,
            self.capacity,
            arch.temporal_d_model // arch.temporal_heads,
            device=device,
            dtype=dtype,
        )
        self.positions = torch.full((self.capacity,), -1, device=device, dtype=torch.long)
        self.next_position = torch.zeros((), device=device, dtype=torch.long)
        self.hidden = torch.zeros(1, 1, arch.d_model, device=device, dtype=dtype)
        self.updates: dict[int, CapturedCall] = {}
        self.decoder: CapturedCall | None = None

    def reset(self) -> None:
        self.positions.fill_(-1)
        self.next_position.zero_()

    def buffers(self) -> tuple[Tensor, ...]:
        return (*self.layers, self.history, self.positions, self.next_position, self.hidden)

    def memory(self) -> KVMemory:
        return KVMemory(self.history, self.positions, self.next_position.reshape(1) - 1, self.window)


def forward_with_kv_cache(model: GPT, features: dict[str, Tensor], observed: Tensor, cache: KVCache) -> Tensor:
    """Append new tokens; each query sees at most L_ctx causal keys at every layer."""
    return forward_tokens_with_kv_cache(model, model.context_tokens(features, observed), cache)


def forward_tokens_with_kv_cache(model: GPT, x: Tensor, cache: KVCache) -> Tensor:
    count = x.shape[1]
    if x.shape[0] != 1 or not 1 <= count <= cache.capacity - cache.window + 1:
        raise ValueError("KV cache input exceeds the prepared batch or update size")
    positions = cache.next_position + torch.arange(count, device=x.device)
    slots = positions % cache.capacity
    cache.positions.index_copy_(0, slots, positions)
    lookback = min(model.trunk.attn_window or cache.window, cache.window)
    valid = (
        (cache.positions[None, :] <= positions[:, None])
        & (cache.positions[None, :] > positions[:, None] - lookback)
        & (cache.positions[None, :] >= 0)
    )
    for module, kv in zip(model.trunk.blocks, cache.layers, strict=True):
        block = cast(Block, module)
        attn = block.attn
        q, k, v = attn.c_attn(rmsnorm(x)).chunk(3, dim=-1)
        shape = (1, count, attn.n_heads, attn.head_dim)
        q = rotate_positions(q.view(shape), positions, attn.rotary).transpose(1, 2)
        k = rotate_positions(k.view(shape), positions, attn.rotary).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        append_kv(kv, slots, torch.stack((k, v)))
        attended = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=valid[None, None])
        attended = attended.transpose(1, 2).reshape(1, count, attn.d_model)
        x = x + block.attn_scale * attn.c_proj(attended)
        x = x + block.mlp_scale * block.mlp(rmsnorm(x))
    hidden = rmsnorm(x)
    history = model.temporal.history_attention
    key, value = history.key_value(F.rms_norm(hidden, (hidden.shape[-1],), eps=1e-6)).chunk(2, dim=-1)
    shape = (1, count, history.n_heads, history.head_dim)
    key = rotate_positions(key.view(shape), positions, history.rotary).transpose(1, 2)
    value = value.view(shape).transpose(1, 2)
    append_kv(cache.history, slots, torch.stack((key, value)))
    cache.hidden.copy_(hidden[:, -1:])
    cache.next_position.add_(count)
    return hidden
