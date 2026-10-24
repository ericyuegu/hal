"""Compiled prefix-cache attention matches the dense position mask on CUDA."""

from collections.abc import Iterator

import pytest
import torch
import torch.nn.functional as F

from hal.inference.cuda_graph import prepared_shape_compilation
from hal.inference.kv_cache import BlockRestartCache
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.action_sequence import short_causal_attention
from hal.models.attention import indexed_prefix_attention
from hal.models.attention import prefix_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention and varlen restart require CUDA")


@pytest.fixture(autouse=True)
def attention_shape_budget() -> Iterator[None]:
    with prepared_shape_compilation(32):
        yield


@pytest.mark.parametrize("queries,capacity", [(1, 65), (21, 259), (127, 259)])
@torch.no_grad()
def test_indexed_history_matches_gather_with_empty_duplicate_and_changing_rows(queries: int, capacity: int) -> None:
    from hal.inference.cuda_graph import CapturedCall

    torch.manual_seed(83)
    rows = torch.tensor([4, 1, 4, 0, 3, 3], device="cuda")
    lengths = torch.tensor([0, 1, queries - 1, queries, capacity - 1, capacity], device="cuda")
    query = torch.randn(6, 2, queries, 32, device="cuda", dtype=torch.bfloat16)
    key, value = torch.randn(2, 2, 5, capacity, 32, device="cuda", dtype=torch.bfloat16).transpose(1, 2).unbind(0)
    compiled = torch.compile(
        indexed_prefix_attention, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )
    graph = CapturedCall(
        lambda: compiled(query, key, value, lengths, rows),
        (query, key, value, lengths, rows),
        device=torch.device("cuda"),
    )
    for _ in range(2):
        mask = torch.arange(capacity, device="cuda")[None] < lengths[:, None]
        expected = F.scaled_dot_product_attention(
            query, key.index_select(0, rows), value.index_select(0, rows), attn_mask=mask[:, None, None]
        )
        expected = torch.where(mask.any(-1)[:, None, None, None], expected, 0)
        actual = graph((query, key, value, lengths, rows))
        torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)
        lengths.copy_(lengths.flip(0))
        rows.copy_(rows.flip(0))


def test_indexed_history_rejects_unsupported_128_query_path() -> None:
    query = torch.zeros(1, 2, 128, 32, device="cuda")
    key = torch.zeros(1, 2, 259, 32, device="cuda")
    with pytest.raises(ValueError, match="fewer than 128 queries"):
        indexed_prefix_attention(query, key, key, torch.tensor([259], device="cuda"), torch.tensor([0], device="cuda"))


@pytest.mark.parametrize(
    "causal,queries,capacity",
    [(True, 1, 259), (True, 2, 259), (True, 4, 259), (False, 21, 259), (False, 1, 25), (False, 127, 259)],
)
@pytest.mark.parametrize("heads,dimension", [(2, 32), (16, 64)])
@torch.no_grad()
def test_prefix_attention_matches_dense_mask_with_empty_and_changing_lengths(
    causal: bool, queries: int, capacity: int, heads: int, dimension: int
) -> None:
    torch.manual_seed(71)
    lengths = torch.tensor([0, 1, queries - 1, queries, capacity - 1, capacity], device="cuda")
    query = torch.randn(6, queries, heads, dimension, dtype=torch.bfloat16, device="cuda").transpose(1, 2)
    key, value = torch.randn(2, 6, heads, capacity, dimension, dtype=torch.bfloat16, device="cuda")
    compiled = torch.compile(prefix_attention, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    for _ in range(2):
        limits = lengths[:, None]
        if causal:
            limits = limits - queries + torch.arange(1, queries + 1, device="cuda")[None]
        mask = torch.arange(capacity, device="cuda")[None, None] < limits[..., None]
        expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None])
        expected = torch.where(mask.any(-1)[:, None, :, None], expected, 0)
        actual = compiled(query, key, value, lengths, causal=causal)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)
        lengths.copy_(lengths.flip(0))


def test_prefix_attention_preserves_gradients() -> None:
    torch.manual_seed(73)
    query = torch.randn(4, 2, 3, 32, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    key, value = torch.randn(2, 4, 2, 65, 32, dtype=torch.bfloat16, device="cuda").requires_grad_().unbind(0)
    lengths = torch.tensor([0, 1, 64, 65], device="cuda")
    limits = lengths[:, None] - 3 + torch.arange(1, 4, device="cuda")[None]
    mask = torch.arange(65, device="cuda")[None, None] < limits[..., None]
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask[:, None])
    actual = torch.compile(prefix_attention, fullgraph=True, dynamic=False)(query, key, value, lengths, causal=True)
    gradient = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)
    actual_gradients = torch.autograd.grad(actual, (query, key, value), gradient)
    expected_gradients = torch.autograd.grad(expected, (query, key, value), gradient)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=3.2e-2, rtol=3.2e-2)


@pytest.mark.parametrize("indexed", [False, True])
@torch.no_grad()
def test_dynamic_prefix_attention_preserves_distinct_row_lengths(indexed: bool) -> None:
    torch.manual_seed(181)
    query = torch.randn(6, 2, 21, 32, device="cuda", dtype=torch.bfloat16)
    key, value = torch.randn(2, 2, 6, 259, 32, device="cuda", dtype=torch.bfloat16).transpose(1, 2).unbind(0)
    lengths = torch.tensor([0, 1, 20, 21, 258, 259], device="cuda")
    rows = torch.tensor([4, 1, 4, 0, 3, 3], device="cuda")
    operation = indexed_prefix_attention if indexed else prefix_attention
    compiled = torch.compile(operation, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False})
    for _ in range(2):
        actual = (
            compiled(query, key, value, lengths, rows)
            if indexed
            else compiled(query, key, value, lengths, causal=False)
        )
        expected = (
            prefix_attention(query, key[rows], value[rows], lengths, causal=False)
            if indexed
            else prefix_attention(query, key, value, lengths, causal=False)
        )
        torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)
        lengths.copy_(lengths.flip(0))
        rows.copy_(rows.flip(0))


@pytest.mark.parametrize("length", [21, 23, 24])
@torch.no_grad()
def test_aligned_short_attention_matches_unpadded_attention(length: int) -> None:
    torch.manual_seed(79)
    query, key, value = torch.randn(3, 8, 2, length, 64, device="cuda", dtype=torch.bfloat16)
    scores = (query @ key.transpose(-1, -2)).float() * (64**-0.5)
    causal = torch.ones(length, length, dtype=torch.bool, device="cuda").tril()
    weights = scores.masked_fill(~causal, -torch.inf).softmax(-1).to(query.dtype)
    expected = weights @ value
    actual = torch.compile(short_causal_attention, fullgraph=True, dynamic=False)(query, key, value)
    torch.testing.assert_close(actual, expected, atol=1.6e-2, rtol=1.6e-2)


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
