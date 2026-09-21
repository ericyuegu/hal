"""Frozen chronological K/V reference from commit 9c364c7 (before ring storage)."""

from typing import Any
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from hal.training.trunk import apply_rotary_emb
from hal.training.trunk import rmsnorm


def shifting_trunk_forward(
    model: Any,
    tokens: Tensor,
    keys: Tensor,
    values: Tensor,
    hidden: Tensor,
    ctx_pad: Tensor,
    active_steps: Tensor,
    token_positions: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Advance a fixed-width trunk cache by the supplied chronological tokens."""
    batch, steps, width = tokens.shape
    layers, cache_batch, heads, length, head_dim = keys.shape
    if (
        cache_batch != batch
        or values.shape != keys.shape
        or hidden.shape != (batch, length, width)
        or ctx_pad.shape != (batch,)
        or active_steps.shape != (steps, batch)
        or token_positions.shape != (steps, batch)
        or layers != len(model.trunk.blocks)
    ):
        raise ValueError("rolling trunk inputs have incompatible shapes")

    first_attention = cast(Any, model.trunk.blocks[0]).attn
    positions = torch.arange(length, device=tokens.device)
    for step in range(steps):
        active = active_steps[step]
        active_4d = active[:, None, None, None]
        active_token = active[:, None, None]
        next_pad = torch.where(active, (ctx_pad - 1).clamp_min(0), ctx_pad)
        x = tokens[:, step : step + 1]
        inv_freq = first_attention.rotary.inv_freq
        if inv_freq.dtype != torch.float32:
            inv_freq = 1.0 / (
                first_attention.rotary.base ** (torch.arange(0, head_dim, 2, device=tokens.device).float() / head_dim)
            )
        frequencies = token_positions[step, :, None].float() * inv_freq[None]
        cos = frequencies.cos().to(tokens.dtype)[:, None, None]
        sin = frequencies.sin().to(tokens.dtype)[:, None, None]
        next_keys: list[Tensor] = []
        next_values: list[Tensor] = []
        for layer, module in enumerate(model.trunk.blocks):
            block = cast(Any, module)
            attention = block.attn
            normalized = rmsnorm(x)
            query, key, value = attention.c_attn(normalized).split(attention.d_model, dim=-1)
            query = query.view(batch, 1, heads, head_dim)
            key = key.view(batch, 1, heads, head_dim)
            value = value.view(batch, 1, heads, head_dim).transpose(1, 2)
            query = apply_rotary_emb(query, cos, sin).transpose(1, 2)
            key = apply_rotary_emb(key, cos, sin).transpose(1, 2)
            present_key = torch.where(
                active_4d,
                torch.cat((keys[layer, :, :, 1:], key), dim=2),
                keys[layer],
            )
            present_value = torch.where(
                active_4d,
                torch.cat((values[layer, :, :, 1:], value), dim=2),
                values[layer],
            )
            valid = positions[None, :] >= next_pad[:, None]
            attended = F.scaled_dot_product_attention(
                query,
                present_key,
                present_value,
                attn_mask=valid[:, None, None],
            )
            attended = attended.transpose(1, 2).contiguous().view(batch, 1, width)
            candidate = x + block.attn_scale * attention.c_proj(attended)
            candidate = candidate + block.mlp_scale * block.mlp(rmsnorm(candidate))
            x = torch.where(active_token, candidate, x)
            next_keys.append(present_key)
            next_values.append(present_value)
        final = rmsnorm(x)
        hidden = torch.where(
            active_token,
            torch.cat((hidden[:, 1:], final), dim=1),
            hidden,
        )
        keys = torch.stack(next_keys)
        values = torch.stack(next_values)
        ctx_pad = next_pad
    return keys, values, hidden, ctx_pad
