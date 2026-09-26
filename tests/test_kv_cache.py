"""KV cache recurrence, ring eviction, and live cache lifecycle."""

from dataclasses import replace

import pytest
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.data.feature_stats import FeatureStats
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.backends.history_decoder.kv_cache import KVCache
from hal.inference.backends.history_decoder.kv_cache import forward_with_kv_cache
from hal.inference.backends.history_decoder.model import GPT
from hal.inference.backends.history_decoder.model import Architecture
from hal.inference.backends.history_decoder.model import TrainConfig
from hal.inference.backends.history_decoder.policy import _MODEL_FIELDS
from hal.inference.backends.history_decoder.policy import HistoryDecoderPolicy
from hal.inference.warmup import make_warmup_observations
from hal.training.ego_stats import consolidate_key
from hal.training.features import ACTION_CHANNELS
from hal.training.features import ITEM_COLUMNS
from hal.training.features import Context
from hal.training.features import feature_kind
from hal.training.trunk import rmsnorm


def policy() -> HistoryDecoderPolicy:
    arch = replace(
        Architecture(),
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    cfg = TrainConfig(arch=arch)
    stats = {
        consolidate_key(n): FeatureStats(0, 1, -10, 10)
        for n in _MODEL_FIELDS
        if feature_kind(n, ITEM_COLUMNS) not in ("cat", "button", "stick_trigger")
    }
    return HistoryDecoderPolicy(GPT(cfg).eval(), cfg, stats, (), device=torch.device("cpu"), seed=5, compiled=False)


def test_reseed_resets_decode_draws_without_rebuilding_policy() -> None:
    candidate = policy()
    context = Context({}, torch.zeros(1, dtype=torch.long), torch.tensor([0]), torch.tensor([True]))
    candidate.reset_prediction(seed=17)
    candidate._rng.begin(context)
    first = candidate._rng.uniforms("main_stick", [True]).clone()
    candidate.reset_prediction(seed=17)
    candidate._rng.begin(context)
    second = candidate._rng.uniforms("main_stick", [True])
    torch.testing.assert_close(first, second)
    assert candidate.sampling_seed == 17


@pytest.mark.parametrize("update", [1, 2])
@torch.inference_mode()
def test_kv_cache_matches_independent_banded_attention_across_wraps(update: int) -> None:
    torch.manual_seed(19)
    p = policy()
    # Multiple layers are essential: retained states carry older context.
    from hal.inference.backends.history_decoder.model import GPT

    p.cfg = replace(p.cfg, arch=replace(p.cfg.arch, n_layers=3))
    p.model = GPT(p.cfg).eval()
    items = make_warmup_observations(p.spec, p.context_frames, 0, 7, 2)
    stream = p._ingest(items[0], None)
    context = stream.gpu.context(0, 0, True)
    length = 37
    features = {name: value[:, -1:].expand(1, length).clone() for name, value in context.features.items()}
    features["ego_percent"] = torch.rand(1, length)
    features["ego_position_x"] = torch.randn(1, length)
    actions = p.model.codec.quantize(torch.rand(1, length, len(ACTION_CHANNELS)))
    positions = torch.arange(length)
    mask = (positions[:, None] >= positions[None, :]) & (positions[:, None] - positions[None, :] < p.context_frames)
    hidden = p.model.context_tokens(features, actions)
    for block in p.model.trunk.blocks:
        hidden = block(hidden, mask[None, None], torch.zeros(1, dtype=torch.long))
    expected = rmsnorm(hidden)
    cache = KVCache(p.model, update, torch.device("cpu"))
    addresses = tuple(kv.data_ptr() for kv in (*cache.layers, cache.history))
    actual = []
    for start in range(0, length, update):
        stop = min(start + update, length)
        actual.append(
            forward_with_kv_cache(
                p.model, {n: v[:, start:stop] for n, v in features.items()}, actions[:, start:stop], cache
            )
        )
        assert tuple(kv.data_ptr() for kv in (*cache.layers, cache.history)) == addresses
    torch.testing.assert_close(torch.cat(actual, dim=1), expected, atol=2e-6, rtol=2e-5)
    # Cached cross-attention must match a fresh projection of the same retained states.
    cross = p.model.temporal.history_attention
    query = torch.randn(1, 1, p.cfg.arch.temporal_d_model)
    keys, values = cross.project_memory(expected[:, -p.context_frames :])
    expected_query = cross.forward_projected(
        query,
        keys,
        values,
        torch.zeros(1, dtype=torch.long),
        torch.tensor([[p.context_frames - 1]]),
        offsets_per_prefix=1,
    )
    torch.testing.assert_close(
        cross.forward_with_kv_cache(query, cache.memory()), expected_query, atol=2e-6, rtol=2e-5
    )
    assert cache.next_position.item() == length
    decoder = p.model.temporal
    decode_args = (actions[:, -1], p.model.head_offsets[:4], torch.tensor([20.0]), torch.tensor([True]))
    _, projected_logits = decoder.sample_indices_with_logits(
        expected[:, -p.context_frames :], *decode_args, argmax=True, forced_prefix=actions[:, -4:]
    )
    _, cached_logits = decoder.sample_indices_with_logits(
        cache.hidden, *decode_args, argmax=True, forced_prefix=actions[:, -4:], history=cache.memory()
    )
    for actual_logits, expected_logits in zip(cached_logits, projected_logits, strict=True):
        torch.testing.assert_close(actual_logits, expected_logits, atol=2e-5, rtol=2e-4)
    # Before eviction this is also ordinary full causal-prefix inference.
    prefix = p.model.forward_dense(
        {n: v[:, :8] for n, v in features.items()}, torch.zeros(1, dtype=torch.long), actions[:, :8]
    )
    torch.testing.assert_close(expected[:, :8], prefix, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("update", [1, 2, 4])
@torch.inference_mode()
def test_kv_cache_prediction_reset_and_conditioning(update: int) -> None:
    p = policy()
    p.history_mode = "kv_cache"
    p.kv_update_frames = update
    p.prepare_prediction(RuntimeConfig(1, (2,)), 4, 2)
    inputs = make_warmup_observations(p.spec, p.context_frames, 0, 7, 2)
    request = PredictionRequest(0, 1, 0, 7, inputs, (NEUTRAL_CONTROLLER_ACTION,) * 2)
    first_plan = p.predict((request,))[0]
    first_cache = p._prediction_streams[0].cache
    assert first_cache is not None
    # Settings change decoder conditioning; they must preserve the trunk cache.
    frame8 = replace(inputs[-1], frame_id=8, reset=False, temperature=0.8, desired_return=None)
    p.predict((PredictionRequest(0, 1, 1, 8, (frame8,), request.fixed_actions),))
    assert p._prediction_streams[0].cache is first_cache
    frame9 = replace(inputs[-1], frame_id=9, reset=False, temperature=1.1, desired_return=40.0)
    p.predict((PredictionRequest(0, 1, 2, 9, (frame9,), request.fixed_actions),))
    assert p._prediction_streams[0].cache is first_cache
    frame100 = replace(inputs[0], frame_id=100, reset=True)
    p.predict((PredictionRequest(0, 2, 0, 100, (frame100,), request.fixed_actions),))
    assert p._prediction_streams[0].cache is first_cache
    assert first_cache.next_position.item() == 1
    p.reset_prediction()
    assert p.predict((request,))[0] == first_plan
    with pytest.raises(ValueError, match="identity changed"):
        p.predict(
            (PredictionRequest(0, 1, 1, 8, (replace(frame8, player_identity="PLATINUM"),), request.fixed_actions),)
        )


@pytest.mark.parametrize("update", [1, 2, 4])
@torch.inference_mode()
def test_token_staging_matches_window_features_and_actions(update: int) -> None:
    from hal.inference.backends.history_decoder.gpu_history import GpuTokenHistory

    p = policy()
    first = make_warmup_observations(p.spec, p.context_frames, 0, 7, 2)[0]
    stream = p._ingest(first, None)
    tokens = GpuTokenHistory(stream.history, p.model.codec, p.device, update)
    action = torch.zeros(len(ACTION_CHANNELS)).numpy()
    for frame in range(1, 30):
        observation = {**first.observation, "p1_percent": float(frame), "p1_position_x": float(frame % 5)}
        stream = p._ingest(replace(first, frame_id=frame, reset=False, observation=observation), stream)
        tokens.push(action)
        actual = tokens.context(0, 0, False)
        expected = stream.gpu.context(0, 0, False)
        count = min(frame, update)
        for name, value in actual.features.items():
            torch.testing.assert_close(value[:, -count:], expected.features[name][:, -count:], atol=0, rtol=0)
        torch.testing.assert_close(tokens.action_indices()[:, -count:], stream.gpu.action_indices()[:, -count:])


@torch.inference_mode()
def test_four_frame_prediction_uses_one_cache_update_and_handles_partial_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = policy()
    candidate.history_mode = "kv_cache"
    candidate.kv_update_frames = 4
    candidate.prepare_prediction(RuntimeConfig(1, (2,)), 8, 4)
    updates = []
    advance = candidate._advance_stream

    def record(stream: object) -> None:
        assert hasattr(stream, "pending_frames")
        count = stream.pending_frames
        if count:
            updates.append(count)
        advance(stream)

    monkeypatch.setattr(candidate, "_advance_stream", record)
    prefix = (NEUTRAL_CONTROLLER_ACTION,) * 4
    initial = make_warmup_observations(candidate.spec, 1, 0, 0, 2)
    candidate.predict((PredictionRequest(0, 1, 0, 0, initial, prefix),))
    assert updates == [1]
    updates.clear()
    four = tuple(replace(item, reset=False) for item in make_warmup_observations(candidate.spec, 4, 0, 4, 2))
    plan = candidate.predict((PredictionRequest(0, 1, 1, 4, four, prefix),))[0]
    assert updates == [4]
    assert len(plan.actions) == 4
    updates.clear()
    two = tuple(replace(item, reset=False) for item in make_warmup_observations(candidate.spec, 2, 0, 6, 2))
    candidate.predict((PredictionRequest(0, 1, 2, 6, two, prefix),))
    assert updates == [2]


@pytest.mark.integration
@torch.inference_mode()
def test_cuda_graph_kv_cache_matches_eager_across_wrap_reset_and_settings() -> None:
    import os

    if os.environ.get("HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 on the production GPU")
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for streaming graph qualification")
    torch.manual_seed(31)
    template = policy()
    compiled = HistoryDecoderPolicy(
        template.model.to("cuda"),
        template.cfg,
        template.stats,
        (),
        device=torch.device("cuda"),
        seed=5,
        compiled=True,
        history_mode="kv_cache",
    )
    import copy

    eager = HistoryDecoderPolicy(
        copy.deepcopy(compiled.model),
        compiled.cfg,
        compiled.stats,
        (),
        device=torch.device("cuda"),
        seed=5,
        compiled=False,
        history_mode="kv_cache",
    )
    runtime = RuntimeConfig(1, (2,))
    compiled.prepare_prediction(runtime, 4, 2)
    eager.prepare_prediction(runtime, 4, 2)
    initial = make_warmup_observations(compiled.spec, compiled.context_frames, 0, 7, 2)[0]
    cache = compiled._caches[0]
    addresses = tuple(value.data_ptr() for value in cache.buffers())
    with torch.compiler.set_stance("fail_on_recompile"):
        for frame in range(45):
            item = replace(
                initial,
                frame_id=frame,
                reset=frame in (0, 24),
                observation={**initial.observation, "p1_percent": float(frame)},
                desired_return=None if frame % 3 == 0 else 20.0,
                temperature=0.8 if frame % 2 else 1.1,
            )
            request = PredictionRequest(
                0,
                1 if frame < 24 else 2,
                frame,
                frame,
                (item,),
                (NEUTRAL_CONTROLLER_ACTION,) * 2,
            )
            compiled.predict((request,))
            eager.predict((request,))
            assert tuple(value.data_ptr() for value in cache.buffers()) == addresses
            other = eager._caches[0]
            torch.testing.assert_close(cache.next_position, other.next_position)
            torch.testing.assert_close(cache.positions, other.positions)
            torch.testing.assert_close(cache.hidden, other.hidden, atol=0.035, rtol=0.035)
            for actual, expected in zip(cache.layers, other.layers, strict=True):
                torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.035)


def test_kv_cache_rejects_updates_larger_than_the_ring_reserve() -> None:
    from hal.inference.backends.history_decoder.kv_cache import forward_tokens_with_kv_cache

    p = policy()
    cache = KVCache(p.model, 2, torch.device("cpu"))
    with pytest.raises(ValueError, match="update size"):
        forward_tokens_with_kv_cache(p.model, torch.zeros(1, 3, p.cfg.arch.d_model), cache)
    with pytest.raises(ValueError, match="update size"):
        forward_tokens_with_kv_cache(p.model, torch.zeros(2, 1, p.cfg.arch.d_model), cache)
