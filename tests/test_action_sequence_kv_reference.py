"""Streaming KV cache against uncached full-sequence banded attention."""

import pytest
import torch

from hal.inference.kv_cache import KVCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer


def _model(attention_window: int) -> ActionSequenceTransformer:
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=3,
        n_heads=4,
        L_ctx=8,
        attn_window=attention_window,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    return ActionSequenceTransformer(config).eval()


@pytest.mark.parametrize("update_frames", [1, 2, 4])
@pytest.mark.parametrize("attention_window", [4, 8])
@torch.inference_mode()
def test_streaming_cache_matches_uncached_banded_trunk_after_wrap_and_reset(
    update_frames: int,
    attention_window: int,
) -> None:
    """A reused physical ring must match each new sequence's dense trunk."""
    torch.manual_seed(47)
    model = _model(attention_window)
    cache = KVCache(model, update_frames, torch.device("cpu"))
    storage = tuple(value.data_ptr() for value in cache.buffers())

    for length in (23, 19):
        tokens = torch.randn(1, length, model.cfg.d_model)
        expected = model.trunk.forward_dense(tokens, torch.zeros(1, dtype=torch.long))
        actual = []
        for start in range(0, length, update_frames):
            actual.append(forward_tokens_with_kv_cache(model, tokens[:, start : start + update_frames], cache))

        torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=2e-6, rtol=2e-5)
        history = model.temporal.history_attention
        query = torch.randn(1, 1, model.cfg.temporal_d_model)
        projected_key, projected_value = history.project_memory(expected[:, -cache.window :])
        projected = history.forward_projected(
            query,
            projected_key,
            projected_value,
            torch.zeros(1, dtype=torch.long),
            torch.tensor([[cache.window - 1]]),
            offsets_per_prefix=1,
        )
        torch.testing.assert_close(
            history.forward_with_kv_cache(query, cache.memory()), projected, atol=2e-6, rtol=2e-5
        )
        assert tuple(value.data_ptr() for value in cache.buffers()) == storage
        assert cache.next_position.item() == length
        valid = cache.positions[cache.positions >= 0]
        assert sorted(valid.tolist()) == list(range(length - min(length, cache.capacity), length))
        cache.reset()
        assert cache.next_position.item() == 0
        assert bool((cache.positions == -1).all())


def test_unsupported_policy_update_is_rejected() -> None:
    model = _model(8)
    with pytest.raises(ValueError, match="one, two, or four frames"):
        KVCache(model, 8, torch.device("cpu"))
    cache = KVCache(model, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="prepared batch or update size"):
        forward_tokens_with_kv_cache(model, torch.randn(1, 5, model.cfg.d_model), cache)
