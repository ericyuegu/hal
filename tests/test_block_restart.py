"""Growing-window cache and two-cache deployment handoff parity."""

import pytest
import torch

from hal.inference.kv_cache import BlockRestartCache
from hal.inference.kv_cache import DualBlockRestartCache
from hal.inference.kv_cache import KVCache
from hal.inference.kv_cache import forward_tokens_with_block_handoff
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer


def _model() -> ActionSequenceTransformer:
    cfg = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=256,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
    )
    return ActionSequenceTransformer(cfg).eval()


@pytest.mark.parametrize("lengths", [(0, 128), (51, 7), (128, 128)])
@torch.inference_mode()
def test_block_cache_hidden_history_and_logits_match_full_window(lengths: tuple[int, int]) -> None:
    torch.manual_seed(11)
    model = _model()
    cache = BlockRestartCache(model, 2, torch.device("cpu"))
    previous = torch.randn(2, 128, 32)
    current = torch.randn(2, 128, 32)
    cache.restart(previous, torch.tensor(lengths))
    assert cache.cache_seqlens.tolist() == list(lengths)
    model.temporal.configure_live_horizons((12,))
    observed = torch.zeros(2, 4, dtype=torch.long)
    uniforms = torch.rand(12, 4, 2)
    for start in range(0, 128, 4):
        actual = cache.append(current[:, start : start + 4])
        if start not in (0, 60, 124):
            continue
        for row, length in enumerate(lengths):
            tokens = torch.cat(
                (
                    previous[row : row + 1, -length:] if length else previous[row : row + 1, :0],
                    current[row : row + 1, : start + 4],
                ),
                dim=1,
            )
            expected = model.trunk.forward_dense(tokens, torch.zeros(1, dtype=torch.long))
            torch.testing.assert_close(actual[row], expected[0, -4:], atol=2e-6, rtol=2e-5)
            indices = torch.tensor([row])
            memory = cache.memory()
            from hal.models.attention import KVMemory

            row_memory = KVMemory(
                memory.kv.index_select(1, indices),
                memory.positions[row : row + 1],
                memory.query_position[row : row + 1],
                memory.window,
            )
            args = (observed[row : row + 1], tuple(range(1, 13)), torch.tensor([108.0]), torch.tensor([True]))
            _, cached_logits = model.temporal.sample_indices_with_logits(
                actual[row : row + 1, -1:],
                *args,
                argmax=False,
                uniforms=uniforms[..., row : row + 1],
                history=row_memory,
            )
            _, full_logits = model.temporal.sample_indices_with_logits(
                expected, *args, argmax=False, uniforms=uniforms[..., row : row + 1]
            )
            for cached, full in zip(cached_logits, full_logits, strict=True):
                torch.testing.assert_close(cached, full, atol=2e-6, rtol=2e-5)
    assert cache.cache_seqlens.tolist() == [length + 128 for length in lengths]
    with pytest.raises(ValueError, match="exhausted"):
        cache.append(current[:, :1])
    cache.reset(0)
    assert cache.cache_seqlens.tolist() == [0, lengths[1] + 128]


@torch.inference_mode()
def test_two_cache_handoff_matches_reencoding_and_independent_reset() -> None:
    torch.manual_seed(13)
    model = _model()
    cache = DualBlockRestartCache(model, 2, torch.device("cpu"))
    tokens = torch.randn(2, 390, 32)
    episode_start = [0, 0]
    for frame in range(390):
        if frame == 174:
            cache.reset(1)
            episode_start[1] = frame
        actual = cache.append(tokens[:, frame : frame + 1])
        if frame not in (0, 127, 128, 173, 174, 255, 256, 301, 302, 383, 384, 389):
            continue
        for row, reset in enumerate(episode_start):
            block_start = reset + ((frame - reset) // 128) * 128
            context_start = max(reset, block_start - 128)
            full = model.trunk.forward_dense(
                tokens[row : row + 1, context_start : frame + 1], torch.zeros(1, dtype=torch.long)
            )
            torch.testing.assert_close(actual[row], full[0, -1:], atol=2e-6, rtol=2e-5)
            assert cache.active.cache_seqlens[row] == frame + 1 - context_start


@torch.inference_mode()
def test_block_prefill_ignores_nonfinite_previous_episode_padding() -> None:
    torch.manual_seed(37)
    model = _model()
    first = BlockRestartCache(model, 2, torch.device("cpu"))
    second = BlockRestartCache(model, 2, torch.device("cpu"))
    previous = torch.randn(2, 128, 32)
    lengths = torch.tensor([0, 13])
    first.restart(previous, lengths)
    previous[0] = torch.nan
    previous[1, :-13] = torch.nan
    second.restart(previous, lengths)
    current = torch.randn(2, 1, 32)
    torch.testing.assert_close(first.append(current), second.append(current), rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_deployment_handoff_matches_full_window_across_mixed_boundaries(compiled: bool) -> None:
    torch.manual_seed(43)
    model = _model()
    active = KVCache(model, 4, torch.device("cpu"), batch_size=2)
    secondary = KVCache(model, 4, torch.device("cpu"), batch_size=2)
    advance = forward_tokens_with_block_handoff
    if compiled:
        advance = torch.compile(advance, backend="eager", fullgraph=True)
    tokens = torch.randn(2, 276, 32)
    episode_start = [0, 0]
    for frame in range(0, 276, 3):
        if frame == 93:
            active.reset(1)
            secondary.reset(1)
            episode_start[1] = frame
        actual = advance(model, tokens[:, frame : frame + 3], active, secondary)
        if frame not in (0, 123, 126, 129, 219, 222, 252, 255, 258, 273):
            continue
        last = frame + 2
        for row, reset in enumerate(episode_start):
            block_start = reset + ((last - reset) // 128) * 128
            start = max(reset, block_start - 128)
            full = model.trunk.forward_dense(tokens[row : row + 1, start : last + 1], torch.zeros(1, dtype=torch.long))
            torch.testing.assert_close(actual[row, -1], full[0, -1], atol=2e-6, rtol=2e-5)
            assert active.next_position[row] == last + 1 - start
