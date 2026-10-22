"""Compiled prefix-cache attention matches the dense position mask on CUDA."""

import pytest
import torch

from hal.inference.kv_cache import BlockRestartCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention and varlen restart require CUDA")


def _model() -> ActionSequenceTransformer:
    cfg = ActionSequenceConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        L_ctx=256,
        temporal_d_model=64,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=128,
        group_head_dim=16,
        value_hidden_dim=16,
    )
    return ActionSequenceTransformer(cfg).eval().cuda()


@torch.inference_mode()
def test_prefix_cache_matches_masked_cache_through_restart_and_ticks() -> None:
    torch.manual_seed(5)
    model = _model()
    device = torch.device("cuda")
    lengths = torch.tensor([0, 1, 37, 128, 90, 128], device=device)
    batch = len(lengths)
    caches = [BlockRestartCache(model, batch, device, dtype=torch.bfloat16) for _ in range(2)]
    caches[1].cache.prefix = False
    previous = torch.randn(batch, 128, 64, device=device)
    history = model.temporal.history_attention

    def tick(cache: BlockRestartCache, tokens: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden = forward_tokens_with_kv_cache(model, tokens, cache.cache)
        return hidden, history.forward_with_kv_cache(state, cache.memory())

    compiled = torch.compile(tick, fullgraph=True, dynamic=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for cache in caches:
            cache.restart(previous, lengths)
        for layer in range(2):
            torch.testing.assert_close(
                caches[0].cache.layers[layer], caches[1].cache.layers[layer], atol=3e-2, rtol=3e-2
            )
        for step in range(6):
            tokens = torch.randn(batch, 2, 64, device=device)
            state = torch.randn(batch, 16, 64, device=device)
            actual = compiled(caches[0], tokens, state)
            expected = tick(caches[1], tokens, state)
            for value, reference in zip(actual, expected, strict=True):
                assert torch.isfinite(value).all(), step
                torch.testing.assert_close(value.float(), reference.float(), atol=5e-2, rtol=5e-2)
