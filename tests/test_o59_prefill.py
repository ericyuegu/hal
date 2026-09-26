"""Known-prefix temporal prefill against the stepwise live decoder."""

from dataclasses import replace

import pytest
import torch

from hal.inference.backends.history_decoder.kv_cache import KVMemory
from hal.inference.backends.history_decoder.model import GPT
from hal.inference.backends.history_decoder.model import Architecture
from hal.inference.backends.history_decoder.model import TrainConfig
from hal.inference.backends.history_decoder.model import decoder_rmsnorm
from hal.training.controller_codec import CONTROLLER_GROUP_COUNT
from hal.training.features import ACTION_CHANNELS


def _model() -> GPT:
    arch = replace(
        Architecture(),
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=3,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    model = GPT(TrainConfig(arch=arch)).eval()
    model.temporal.configure_live_horizons((8,))
    return model


@pytest.mark.parametrize("history_kind", ["projected", "cached"])
@pytest.mark.parametrize("prefix_count", [1, 3, 8])
@torch.inference_mode()
def test_parallel_prefix_matches_stepwise_states_and_sampling(history_kind: str, prefix_count: int) -> None:
    torch.manual_seed(31)
    model = _model()
    decoder = model.temporal
    batch = 2 if history_kind == "projected" else 1
    hidden = torch.randn(batch, 8, model.cfg.arch.d_model)
    actions = model.codec.quantize(torch.rand(batch, prefix_count + 1, len(ACTION_CHANNELS)))
    observed = actions[:, 0]
    forced = actions[:, 1:]
    offsets = tuple(range(1, 9))
    returns = torch.tensor([20.0] * batch)
    present = torch.tensor([True] * batch)
    state_bias = decoder._state_bias(decoder_rmsnorm(hidden[:, -1]))
    conditioning = decoder.return_conditioner(returns, present)
    if history_kind == "projected":
        history = decoder._live_history(hidden)
        cached_history = None
    else:
        kv = torch.randn(2, 1, model.cfg.arch.temporal_heads, 8, model.cfg.arch.temporal_d_model // 4)
        cached_history = KVMemory(kv, torch.tensor([17, 18, 11, 12, 13, 14, 15, 16]), torch.tensor([18]), 8)
        history = cached_history

    parallel_states, parallel_cache = decoder._prefill_forced_prefix(
        observed, offsets, forced, state_bias, history, conditioning
    )
    previous = observed
    serial_cache = [None] * len(decoder.blocks)
    serial_states = []
    for depth in range(prefix_count):
        state, serial_cache = decoder._decode_step(
            previous, offsets[depth], state_bias, serial_cache, history, conditioning
        )
        serial_states.append(state)
        previous = forced[:, depth]
    torch.testing.assert_close(
        decoder_rmsnorm(parallel_states), torch.stack(serial_states, dim=1), atol=2e-6, rtol=2e-5
    )
    for parallel, serial in zip(parallel_cache, serial_cache, strict=True):
        assert parallel is not None
        assert serial is not None
        for parallel_part, serial_part in zip(parallel, serial, strict=True):
            torch.testing.assert_close(parallel_part, serial_part, atol=2e-6, rtol=2e-5)

    actual = decoder.sample_indices(
        hidden, observed, offsets, returns, present, argmax=True, forced_prefix=forced, history=cached_history
    )
    expected, _ = decoder.sample_indices_with_logits(
        hidden, observed, offsets, returns, present, argmax=True, forced_prefix=forced, history=cached_history
    )
    torch.testing.assert_close(actual, expected)


@torch.inference_mode()
def test_parallel_prefix_preserves_stochastic_draw_count() -> None:
    torch.manual_seed(37)
    model = _model()
    decoder = model.temporal
    hidden = torch.randn(1, 8, model.cfg.arch.d_model)
    actions = model.codec.quantize(torch.rand(1, 4, len(ACTION_CHANNELS)))
    observed, forced = actions[:, 0], actions[:, 1:]
    returns = torch.tensor([20.0])
    present = torch.tensor([True])
    first = torch.Generator().manual_seed(47)
    second = torch.Generator().manual_seed(47)
    actual = decoder.sample_indices(
        hidden, observed, tuple(range(1, 9)), returns, present, argmax=False, gen=first, forced_prefix=forced
    )
    expected, _ = decoder.sample_indices_with_logits(
        hidden, observed, tuple(range(1, 9)), returns, present, argmax=False, gen=second, forced_prefix=forced
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(first.get_state(), second.get_state())


@torch.inference_mode()
def test_parallel_prefix_uses_absolute_uniform_depth_with_runtime_temperature() -> None:
    torch.manual_seed(41)
    model = _model()
    decoder = model.temporal
    hidden = torch.randn(1, 8, model.cfg.arch.d_model)
    actions = model.codec.quantize(torch.rand(1, 4, len(ACTION_CHANNELS)))
    observed, forced = actions[:, 0], actions[:, 1:]
    uniforms = torch.rand(8, CONTROLLER_GROUP_COUNT, 1)
    returns = torch.tensor([20.0])
    present = torch.tensor([True])
    actual = decoder.sample_indices(
        hidden,
        observed,
        tuple(range(1, 9)),
        returns,
        present,
        argmax=False,
        uniforms=uniforms,
        temperature=torch.tensor(0.9),
        forced_prefix=forced,
    )
    expected, _ = decoder.sample_indices_with_logits(
        hidden,
        observed,
        tuple(range(1, 9)),
        returns,
        present,
        argmax=False,
        uniforms=uniforms,
        temperature=torch.tensor(0.9),
        forced_prefix=forced,
    )
    torch.testing.assert_close(actual, expected)
