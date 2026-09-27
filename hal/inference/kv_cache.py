"""Bounded per-stream KV rings and prepared batch views for cached inference."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from hal.inference.cuda_graph import CapturedCall
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.attention import Block
from hal.models.attention import KVMemory
from hal.models.attention import rmsnorm
from hal.models.attention import rotate_positions

if TYPE_CHECKING:
    from collections.abc import Sequence


@torch.library.custom_op("hal::append_kv_rows", mutates_args=("cache",))
def append_kv_rows(cache: Tensor, slots: Tensor, values: Tensor) -> None:
    """Append distinct absolute positions for each batch row without a Python GPU loop."""
    if cache.shape[1] == 1:
        cache.index_copy_(3, slots[0], values)
        return
    index = slots[None, :, None, :, None].expand_as(values)
    cache.scatter_(3, index, values)


@append_kv_rows.register_fake
def _append_kv_rows_fake(cache: Tensor, slots: Tensor, values: Tensor) -> None:
    pass


class KVCache:
    """Tensor storage for one prepared bucket of independent stream rows."""

    def __init__(
        self,
        model: ActionSequenceTransformer,
        update_frames: int,
        device: torch.device,
        *,
        batch_size: int = 1,
        dtype: torch.dtype | None = None,
    ) -> None:
        if update_frames not in (1, 2, 4):
            raise ValueError("KV cache updates must contain one, two, or four frames")
        if batch_size < 1:
            raise ValueError("KV cache batch size must be positive")
        config = model.cfg
        self.window = config.L_ctx
        self.max_update_frames = update_frames
        self.capacity = self.window + update_frames - 1
        self.batch_size = batch_size
        if dtype is None:
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.layers = tuple(
            torch.zeros(
                2,
                batch_size,
                config.n_heads,
                self.capacity,
                config.d_model // config.n_heads,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.n_layers)
        )
        self.history = torch.zeros(
            2,
            batch_size,
            config.temporal_heads,
            self.capacity,
            config.temporal_d_model // config.temporal_heads,
            device=device,
            dtype=dtype,
        )
        self.positions = torch.full((batch_size, self.capacity), -1, device=device, dtype=torch.long)
        self.next_position = torch.zeros((batch_size,), device=device, dtype=torch.long)
        self.hidden = torch.zeros(batch_size, 1, config.d_model, device=device, dtype=dtype)
        self.updates: dict[int, CapturedCall] = {}
        self.decoder: CapturedCall | None = None

    def reset(self, row: int | None = None) -> None:
        if row is None:
            self.positions.fill_(-1)
            self.next_position.zero_()
            self.hidden.zero_()
            return
        if not 0 <= row < self.batch_size:
            raise ValueError("KV cache row is out of range")
        self.positions[row].fill_(-1)
        self.next_position[row].zero_()
        self.hidden[row].zero_()

    def buffers(self) -> tuple[Tensor, ...]:
        return (*self.layers, self.history, self.positions, self.next_position, self.hidden)

    def memory(self) -> KVMemory:
        return KVMemory(self.history, self.positions, self.next_position - 1, self.window)

    def row_view(self, row: int) -> KVCache:
        """Return a batch-one view of persistent storage without copying its KV tensors."""
        if not 0 <= row < self.batch_size:
            raise ValueError("KV cache row is out of range")
        view = object.__new__(KVCache)
        view.window = self.window
        view.max_update_frames = self.max_update_frames
        view.capacity = self.capacity
        view.batch_size = 1
        view.layers = tuple(layer[:, row : row + 1] for layer in self.layers)
        view.history = self.history[:, row : row + 1]
        view.positions = self.positions[row : row + 1]
        view.next_position = self.next_position[row : row + 1]
        view.hidden = self.hidden[row : row + 1]
        view.updates = {}
        view.decoder = None
        return view


class KVCachePool:
    """Persistent stream rows plus one prepared scratch cache per batch bucket."""

    def __init__(
        self,
        model: ActionSequenceTransformer,
        stream_capacity: int,
        device: torch.device,
        *,
        max_update_frames: int = 4,
        dtype: torch.dtype | None = None,
    ) -> None:
        if stream_capacity < 1:
            raise ValueError("stream capacity must be positive")
        self.capacity = stream_capacity
        self.storage = KVCache(model, max_update_frames, device, batch_size=stream_capacity, dtype=dtype)
        self.scratch: dict[int, KVCache] = {}
        if stream_capacity > 1:
            self.scratch[1] = KVCache(model, max_update_frames, device, batch_size=1, dtype=dtype)
        bucket = 2
        while bucket // 2 < stream_capacity:
            self.scratch[bucket] = KVCache(model, max_update_frames, device, batch_size=bucket, dtype=dtype)
            bucket *= 2

    def reset(self, row: int) -> None:
        self.storage.reset(row)

    def row_view(self, row: int) -> KVCache:
        return self.storage.row_view(row)

    def gather(self, rows: Sequence[int], bucket: int, *, direct: bool = True) -> KVCache:
        """Copy only ready rows into their prepared bucket; dummy rows stay invalid."""
        selected = tuple(rows)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("ready cache rows must be non-empty and unique")
        if any(row < 0 or row >= self.capacity for row in selected):
            raise ValueError("ready cache row is out of range")
        if bucket < len(selected) or bucket & (bucket - 1):
            raise ValueError("batch bucket must be a covering power of two")
        if bucket == 1 and direct:
            return self.row_view(selected[0])
        try:
            scratch = self.scratch[bucket]
        except KeyError as error:
            raise ValueError("batch bucket was not prepared") from error
        indices = torch.tensor(selected, device=self.storage.positions.device, dtype=torch.long)
        count = len(selected)
        for source, target in zip(self.storage.layers, scratch.layers, strict=True):
            target[:, :count].copy_(source.index_select(1, indices))
        scratch.history[:, :count].copy_(self.storage.history.index_select(1, indices))
        scratch.positions[:count].copy_(self.storage.positions.index_select(0, indices))
        scratch.next_position[:count].copy_(self.storage.next_position.index_select(0, indices))
        scratch.hidden[:count].copy_(self.storage.hidden.index_select(0, indices))
        if count < bucket:
            scratch.positions[count:].fill_(-1)
            scratch.next_position[count:].zero_()
            scratch.hidden[count:].zero_()
        return scratch

    def scatter(self, rows: Sequence[int], cache: KVCache) -> None:
        """Commit only real ready rows; dummy rows cannot change persistent state."""
        selected = tuple(rows)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("ready cache rows must be non-empty and unique")
        if any(row < 0 or row >= self.capacity for row in selected):
            raise ValueError("ready cache row is out of range")
        if cache.batch_size < len(selected):
            raise ValueError("cache bucket is smaller than the ready batch")
        if (
            len(selected) == 1
            and cache.positions.data_ptr() == self.storage.positions[selected[0] : selected[0] + 1].data_ptr()
        ):
            return
        indices = torch.tensor(selected, device=self.storage.positions.device, dtype=torch.long)
        count = len(selected)
        for target, source in zip(self.storage.layers, cache.layers, strict=True):
            target.index_copy_(1, indices, source[:, :count])
        self.storage.history.index_copy_(1, indices, cache.history[:, :count])
        self.storage.positions.index_copy_(0, indices, cache.positions[:count])
        self.storage.next_position.index_copy_(0, indices, cache.next_position[:count])
        self.storage.hidden.index_copy_(0, indices, cache.hidden[:count])


def forward_with_kv_cache(
    model: ActionSequenceTransformer,
    features: dict[str, Tensor],
    observed: Tensor,
    cache: KVCache,
) -> Tensor:
    """Append new model tokens and retain at most the declared context."""
    return forward_tokens_with_kv_cache(model, model.context_tokens(features, observed), cache)


def forward_tokens_with_kv_cache(model: ActionSequenceTransformer, x: Tensor, cache: KVCache) -> Tensor:
    if x.ndim != 3 or x.shape[0] != cache.batch_size or not 1 <= x.shape[1] <= cache.max_update_frames:
        raise ValueError("KV cache input exceeds the prepared batch or update size")
    batch, count, _ = x.shape
    positions = cache.next_position[:, None] + torch.arange(count, device=x.device)[None]
    slots = positions % cache.capacity
    cache.positions.scatter_(1, slots, positions)
    lookback = min(model.trunk.attn_window or cache.window, cache.window)
    valid = (
        (cache.positions[:, None, :] <= positions[:, :, None])
        & (cache.positions[:, None, :] > positions[:, :, None] - lookback)
        & (cache.positions[:, None, :] >= 0)
    )
    for module, kv in zip(model.trunk.blocks, cache.layers, strict=True):
        block = cast(Block, module)
        attn = block.attn
        q, k, v = attn.c_attn(rmsnorm(x)).chunk(3, dim=-1)
        shape = (batch, count, attn.n_heads, attn.head_dim)
        q = rotate_positions(q.view(shape), positions, attn.rotary).transpose(1, 2)
        k = rotate_positions(k.view(shape), positions, attn.rotary).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        append_kv_rows(kv, slots, torch.stack((k, v)))
        attended = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=valid[:, None])
        attended = attended.transpose(1, 2).reshape(batch, count, attn.d_model)
        x = x + block.attn_scale * attn.c_proj(attended)
        x = x + block.mlp_scale * block.mlp(rmsnorm(x))
    hidden = rmsnorm(x)
    history = model.temporal.history_attention
    key, value = history.key_value(F.rms_norm(hidden, (hidden.shape[-1],), eps=1e-6)).chunk(2, dim=-1)
    shape = (batch, count, history.n_heads, history.head_dim)
    key = rotate_positions(key.view(shape), positions, history.rotary).transpose(1, 2)
    value = value.view(shape).transpose(1, 2)
    append_kv_rows(cache.history, slots, torch.stack((key, value)))
    cache.hidden.copy_(hidden[:, -1:])
    cache.next_position.add_(count)
    return hidden
