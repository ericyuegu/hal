"""Independent cached rows must match serial streams through wraps and reuse."""

import pytest
import torch
import torch.nn.functional as F

from hal.inference.kv_cache import KVCache
from hal.inference.kv_cache import KVCachePool
from hal.inference.kv_cache import forward_tokens_with_kv_cache
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.wire import ACTION_DIM


def _model() -> ActionSequenceTransformer:
    torch.manual_seed(47)
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=8,
        attn_window=8,
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


@torch.inference_mode()
def test_batched_cache_matches_serial_rows_after_wrap_reset_and_reorder() -> None:
    model = _model()
    device = torch.device("cpu")
    batched = KVCachePool(model, 5, device)
    serial = [KVCache(model, 4, device) for _ in range(5)]
    generator = torch.Generator().manual_seed(73)
    waves = (
        ((4, 1, 3), 4),
        ((3, 4), 2),
        ((1,), 1),
        ((0, 3, 4, 1), 4),
        ((2, 4), 1),
        ((4, 3, 0), 2),
    )
    for wave in range(4):
        for rows, count in waves:
            if wave == 2 and rows == (4, 1, 3):
                batched.reset(3)
                serial[3].reset()
            tokens = torch.randn(len(rows), count, model.cfg.d_model, generator=generator)
            bucket = 1 << (len(rows) - 1).bit_length()
            cache = batched.gather(rows, bucket)
            bucket_tokens = torch.zeros(bucket, count, model.cfg.d_model)
            bucket_tokens[: len(rows)] = tokens
            actual = forward_tokens_with_kv_cache(model, bucket_tokens, cache)
            expected = torch.cat(
                [
                    forward_tokens_with_kv_cache(model, tokens[index : index + 1], serial[row])
                    for index, row in enumerate(rows)
                ]
            )
            torch.testing.assert_close(actual[: len(rows)], expected, atol=2e-6, rtol=2e-5)
            query = torch.randn(len(rows), 2, model.cfg.temporal_d_model, generator=generator)
            bucket_query = torch.zeros(bucket, 2, model.cfg.temporal_d_model)
            bucket_query[: len(rows)] = query
            attended = model.temporal.history_attention.forward_with_kv_cache(bucket_query, cache.memory())
            reference = torch.cat(
                [
                    model.temporal.history_attention.forward_with_kv_cache(
                        query[index : index + 1], serial[row].memory()
                    )
                    for index, row in enumerate(rows)
                ]
            )
            torch.testing.assert_close(attended[: len(rows)], reference, atol=2e-6, rtol=2e-5)
            batched.scatter(rows, cache)
            for row, expected_cache in enumerate(serial):
                actual_cache = batched.row_view(row)
                torch.testing.assert_close(actual_cache.positions, expected_cache.positions, rtol=0, atol=0)
                torch.testing.assert_close(actual_cache.next_position, expected_cache.next_position, rtol=0, atol=0)
                torch.testing.assert_close(actual_cache.hidden, expected_cache.hidden, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(actual_cache.history, expected_cache.history, atol=2e-6, rtol=2e-5)
                for actual_kv, expected_kv in zip(actual_cache.layers, expected_cache.layers, strict=True):
                    torch.testing.assert_close(actual_kv, expected_kv, atol=2e-6, rtol=2e-5)
    assert int(batched.storage.next_position.max()) > batched.storage.capacity


def test_covering_buckets_and_invalid_ready_rows() -> None:
    model = _model()
    pool = KVCachePool(model, 3, torch.device("cpu"))
    assert tuple(pool.scratch) == (1, 2, 4)
    assert pool.gather((2,), 1).positions.data_ptr() == pool.row_view(2).positions.data_ptr()
    with pytest.raises(ValueError, match="unique"):
        pool.gather((1, 1), 2)
    with pytest.raises(ValueError, match="out of range"):
        pool.gather((3,), 1)
    with pytest.raises(ValueError, match="covering"):
        pool.gather((1, 2), 1)


@pytest.mark.parametrize("ready_rows", [1, 2, 4, 8, 16, 32])
@torch.inference_mode()
def test_each_prepared_batch_bucket_matches_independent_rows(ready_rows: int) -> None:
    model = _model()
    device = torch.device("cpu")
    pool = KVCachePool(model, 32, device)
    serial = tuple(KVCache(model, 4, device) for _ in range(ready_rows))
    generator = torch.Generator().manual_seed(970 + ready_rows)
    indices = tuple(reversed(range(ready_rows)))
    for count in (1, 2, 4):
        tokens = torch.randn(ready_rows, count, model.cfg.d_model, generator=generator)
        cache = pool.gather(indices, ready_rows)
        actual = forward_tokens_with_kv_cache(model, tokens, cache)
        expected = torch.cat(
            [forward_tokens_with_kv_cache(model, tokens[row : row + 1], serial[row]) for row in range(ready_rows)]
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        pool.scatter(indices, cache)
        for row, control in zip(indices, serial, strict=True):
            candidate = pool.row_view(row)
            torch.testing.assert_close(candidate.positions, control.positions, rtol=0, atol=0)
            torch.testing.assert_close(candidate.next_position, control.next_position, rtol=0, atol=0)
            torch.testing.assert_close(candidate.hidden, control.hidden, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_dummy_rows_never_change_persistent_state() -> None:
    model = _model()
    pool = KVCachePool(model, 3, torch.device("cpu"))
    first = pool.gather((0, 2, 1), 4)
    forward_tokens_with_kv_cache(model, torch.randn(4, 4, model.cfg.d_model), first)
    pool.scatter((0, 2, 1), first)
    before = tuple(value.clone() for value in pool.row_view(1).buffers())
    second = pool.gather((2, 0), 4)
    assert bool((second.positions[2:] == -1).all())
    forward_tokens_with_kv_cache(model, torch.randn(4, 2, model.cfg.d_model), second)
    pool.scatter((2, 0), second)
    for old, new in zip(before, pool.row_view(1).buffers(), strict=True):
        torch.testing.assert_close(new, old, rtol=0, atol=0)


def test_cache_rejects_unsupported_update_shape() -> None:
    model = _model()
    with pytest.raises(ValueError, match="one, two, or four frames"):
        KVCache(model, 3, torch.device("cpu"))
    cache = KVCache(model, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="prepared batch or update size"):
        forward_tokens_with_kv_cache(model, torch.randn(1, 5, model.cfg.d_model), cache)


@torch.inference_mode()
def test_fp32_batched_teacher_forced_decoder_logits_match_serial_after_eviction() -> None:
    model = _model()
    model.temporal.configure_live_horizons((4,))
    device = torch.device("cpu")
    pool = KVCachePool(model, 2, device)
    serial = (KVCache(model, 4, device), KVCache(model, 4, device))
    generator = torch.Generator().manual_seed(912)
    observed = model.codec.quantize(torch.zeros(2, ACTION_DIM))
    forced = observed[:, None].expand(2, 4, observed.shape[-1])
    returns = torch.tensor([20.0, 30.0])
    present = torch.ones(2, dtype=torch.bool)
    for _ in range(4):
        tokens = torch.randn(2, 4, model.cfg.d_model, generator=generator)
        cache = pool.gather((1, 0), 2)
        forward_tokens_with_kv_cache(model, tokens, cache)
        pool.scatter((1, 0), cache)
        for index, row in enumerate((1, 0)):
            forward_tokens_with_kv_cache(model, tokens[index : index + 1], serial[row])
        cache = pool.gather((1, 0), 2)
        _, batch_logits = model.temporal.sample_indices_with_logits(
            cache.hidden,
            observed,
            (1, 2, 3, 4),
            returns,
            present,
            argmax=True,
            forced_prefix=forced,
            history=cache.memory(),
        )
        for index, row in enumerate((1, 0)):
            _, one_logits = model.temporal.sample_indices_with_logits(
                serial[row].hidden,
                observed[index : index + 1],
                (1, 2, 3, 4),
                returns[index : index + 1],
                present[index : index + 1],
                argmax=True,
                forced_prefix=forced[index : index + 1],
                history=serial[row].memory(),
            )
            for actual, expected in zip(batch_logits, one_logits, strict=True):
                torch.testing.assert_close(actual[index : index + 1], expected, atol=2e-5, rtol=2e-4)
    assert int(pool.storage.next_position.max()) > model.cfg.L_ctx


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ready_rows", [1, 2, 4, 8, 16, 32])
@torch.inference_mode()
def test_cuda_bf16_batched_conditional_logits_match_serial_before_and_after_eviction(ready_rows: int) -> None:
    model = _model().to(device="cuda", dtype=torch.bfloat16)
    model.temporal.configure_live_horizons((4,))
    device = torch.device("cuda")
    pool = KVCachePool(model, ready_rows, device)
    serial = tuple(KVCache(model, 4, device) for _ in range(ready_rows))
    generator = torch.Generator().manual_seed(73)
    observed = model.codec.quantize(torch.zeros(ready_rows, ACTION_DIM, device=device))
    forced = observed[:, None].expand(ready_rows, 4, observed.shape[-1])
    returns = torch.linspace(20.0, 30.0, ready_rows, device=device)
    present = torch.ones(ready_rows, dtype=torch.bool, device=device)
    divergences = []
    indices = tuple(reversed(range(ready_rows)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(8):
            tokens = torch.randn(ready_rows, 4, model.cfg.d_model, generator=generator).to(device, torch.bfloat16)
            cache = pool.gather(indices, ready_rows)
            forward_tokens_with_kv_cache(model, tokens, cache)
            pool.scatter(indices, cache)
            for index, row in enumerate(indices):
                forward_tokens_with_kv_cache(model, tokens[index : index + 1], serial[row])
            cache = pool.gather(indices, ready_rows)
            _, batch_logits = model.temporal.sample_indices_with_logits(
                cache.hidden,
                observed,
                (1, 2, 3, 4),
                returns,
                present,
                argmax=True,
                forced_prefix=forced,
                history=cache.memory(),
            )
            for index, row in enumerate(indices):
                _, one_logits = model.temporal.sample_indices_with_logits(
                    serial[row].hidden,
                    observed[index : index + 1],
                    (1, 2, 3, 4),
                    returns[index : index + 1],
                    present[index : index + 1],
                    argmax=True,
                    forced_prefix=forced[index : index + 1],
                    history=serial[row].memory(),
                )
                for actual, expected in zip(batch_logits, one_logits, strict=True):
                    p = F.softmax(expected[0].float(), dim=-1)
                    q = F.softmax(actual[index].float(), dim=-1)
                    kl = torch.where(p > 0, p * (p.log() - q.log()), 0.0).sum(dim=-1)
                    divergences.extend(kl.tolist())
    assert all(value >= -1e-6 and value < float("inf") for value in divergences)
    measured = torch.tensor(divergences)
    mean = measured.mean().item()
    p99 = torch.quantile(measured, 0.99).item()
    print(f"BF16 B{ready_rows}/serial conditional KL: mean={mean:.8g}, p99={p99:.8g}, n={measured.numel()}")
    assert mean <= 5e-4
    assert p99 <= 5e-3
