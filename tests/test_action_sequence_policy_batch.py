"""Ready streams use one cached model call while preserving stream identity."""

import os
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import melee.gamestate
import numpy as np
import pytest
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.eval.observations import flatten_live_frame
from hal.eval.observations import policy_input_from_frame
from hal.inference.action_sequence_artifact import REQUIRED_OBSERVATION_FIELDS
from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import ActionPlan
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.cuda_graph import count_compilation_starts
from hal.inference.kv_cache import KVCache
from hal.inference.observation_history import ObservationHistory
from hal.inference.warmup import make_warmup_observations
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.attention import KVMemory
from hal.models.controller_codec import DiscreteControllerCodec
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind
from hal.sim.session import canonical_frame
from hal.wire import ACTION_DIM


def _assert_plans_match(actual: Sequence[ActionPlan], expected: Sequence[ActionPlan]) -> None:
    actual_by_id = {plan.stream_id: plan for plan in actual}
    expected_by_id = {plan.stream_id: plan for plan in expected}
    assert actual_by_id.keys() == expected_by_id.keys()
    for stream_id, plan in actual_by_id.items():
        reference = expected_by_id[stream_id]
        assert replace(plan, state_value=0.0) == replace(reference, state_value=0.0)
        assert plan.state_value == pytest.approx(reference.state_value, abs=2e-6, rel=2e-5)


def _policy(
    capacity: int,
    *,
    model: ActionSequenceTransformer | None = None,
    device: torch.device | None = None,
    compiled: bool = False,
    cuda_graphs: bool = False,
    history_mode: Literal["window", "kv_cache"] = "kv_cache",
    delay: int = 0,
    horizon: int = 4,
    prefix: int = 0,
) -> ActionSequencePolicy:
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(47)
    config = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=2,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    stats: dict[str, FeatureStats] = {}
    for name in REQUIRED_OBSERVATION_FIELDS:
        relative = (
            f"ego_{name[3:]}" if name.startswith("p1_") else f"opp_{name[3:]}" if name.startswith("p2_") else name
        )
        if feature_kind(relative, ITEM_COLUMNS) not in ("cat", "button", "stick_trigger"):
            stats[consolidate_key(relative)] = FeatureStats(0.0, 1.0, -10.0, 10.0)
    spec = PolicySpec("small 059", "hal.action_sequence.test", REQUIRED_OBSERVATION_FIELDS, (0, 2, 3))
    policy = ActionSequencePolicy(
        (ActionSequenceTransformer(config).eval().to(device) if model is None else model),
        stats,
        (),
        spec=spec,
        checkpoint_sha256="a" * 64,
        return_p90=20.0,
        capability_version=1,
        device=device,
        seed=5,
        compiled=compiled,
        history_mode=history_mode,
        kv_update_frames=4,
        kv_cuda_graphs=cuda_graphs,
    )
    policy.prepare_prediction(RuntimeConfig(capacity, (delay,)), horizon, prefix)
    return policy


def test_cached_prepared_profile_rejects_wrong_identity_or_uncaptured_shape() -> None:
    policy = _policy(2)
    profile = PreparedInferenceProfile("test", "a" * 64, "kv_cache", 4, 0, (1, 2, 4), 2)
    policy.validate_prepared_profile(profile)
    with pytest.raises(ValueError, match="differs"):
        policy.validate_prepared_profile(replace(profile, checkpoint_sha256="b" * 64))
    with pytest.raises(ValueError, match="update shape"):
        policy.validate_prepared_profile(replace(profile, update_shapes=(1, 2, 4, 8)))


def test_cached_preparation_warms_delivered_action_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    shapes: list[tuple[int, ...]] = []
    original = DiscreteControllerCodec.dequantize

    def record(self: DiscreteControllerCodec, indices: torch.Tensor) -> torch.Tensor:
        shapes.append(tuple(indices.shape))
        return original(self, indices)

    monkeypatch.setattr(DiscreteControllerCodec, "dequantize", record)
    policy = _policy(3, delay=2, horizon=8, prefix=3)

    assert shapes == [(1, 8, 4), (2, 8, 4), (4, 8, 4)]
    assert not policy._prediction_streams
    assert policy._rng.state() == ()
    before = torch.get_rng_state().clone()
    policy.prepare_prediction(RuntimeConfig(3, (2,)), 8, 3)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert policy._rng.state() == ()


def test_failed_cached_repreparation_cannot_be_admitted(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy(2)
    profile = PreparedInferenceProfile("test", "a" * 64, "kv_cache", 4, 0, (1, 2, 4), 2)
    policy.validate_prepared_profile(profile)

    def fail() -> None:
        raise RuntimeError("capture failed")

    monkeypatch.setattr(policy, "_prepare_kv_cache", fail)
    with pytest.raises(RuntimeError, match="capture failed"):
        policy.prepare_prediction(RuntimeConfig(2, (0,)), 4, 0)
    with pytest.raises(ValueError, match="has not been prepared"):
        policy.validate_prepared_profile(profile)


def _request(
    policy: ActionSequencePolicy, stream_id: int, source: int, count: int, sequence: int
) -> PredictionRequest:
    observations = make_warmup_observations(policy.spec, count, stream_id, source, 0, reset_first=sequence == 0)
    observations = tuple(
        replace(item, desired_return=20.0 + stream_id % 3, temperature=0.8 if stream_id == 42 else 1.1)
        for item in observations
    )
    return PredictionRequest(stream_id, 1, sequence, source, observations, ())


def _conditioned_request(
    policy: ActionSequencePolicy,
    stream_id: int,
    source: int,
    count: int,
    sequence: int,
    *,
    identity: str,
    desired_return: float | None,
    temperature: float,
    prefix: int = 0,
) -> PredictionRequest:
    request = _request(policy, stream_id, source, count, sequence)
    observations = tuple(
        replace(item, player_identity=identity, desired_return=desired_return, temperature=temperature)
        for item in request.observations
    )
    return replace(request, observations=observations, fixed_actions=(NEUTRAL_CONTROLLER_ACTION,) * prefix)


def test_ready_cached_streams_share_gpu_calls_and_keep_stream_rng() -> None:
    batched = _policy(4)
    serial = _policy(4)
    calls: list[tuple[str, int, int]] = []
    trunk = batched._kv_trunk
    decoder = batched._kv_decoder

    def record_trunk(features: dict[str, torch.Tensor], observed: torch.Tensor, cache: KVCache) -> torch.Tensor:
        calls.append(("trunk", observed.shape[0], observed.shape[1]))
        return trunk(features, observed, cache)

    def record_decoder(
        hidden: torch.Tensor,
        memory: KVMemory,
        observed: torch.Tensor,
        uniforms: torch.Tensor,
        forced: torch.Tensor,
        return_value: torch.Tensor,
        condition_present: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        calls.append(("decoder", hidden.shape[0], 1))
        return decoder(hidden, memory, observed, uniforms, forced, return_value, condition_present, temperature)

    batched._kv_trunk = record_trunk
    batched._kv_decoder = record_decoder
    first = (_request(batched, 42, 3, 4, 0), _request(batched, 7, 3, 4, 0))
    actual = batched.predict(first)
    expected = tuple(serial.predict((request,))[0] for request in reversed(first))
    assert {plan.stream_id: plan.actions for plan in actual} == {plan.stream_id: plan.actions for plan in expected}
    assert calls == [("trunk", 2, 4), ("decoder", 2, 1)]
    assert [plan.actions[0].target_frame for plan in actual] == [4, 4]
    assert batched._rng.state() == serial._rng.state()

    calls.clear()
    next_requests = (_request(batched, 7, 6, 3, 1), _request(batched, 42, 6, 3, 1))
    actual = batched.predict(next_requests)
    expected = tuple(serial.predict((request,))[0] for request in reversed(next_requests))
    assert {plan.stream_id: plan.actions for plan in actual} == {plan.stream_id: plan.actions for plan in expected}
    assert calls == [("trunk", 2, 2), ("trunk", 2, 1), ("decoder", 2, 1)]
    assert batched._rng.state() == serial._rng.state()
    for stream_id in (7, 42):
        cache = batched._prediction_streams[stream_id].cache
        assert cache is not None and cache.next_position.item() == 7


@pytest.mark.parametrize("delay,horizon,prefix", [(0, 4, 0), (0, 4, 2), (2, 8, 3), (3, 8, 4)])
def test_ready_rows_keep_each_fixed_prefix_shape_and_sampling_state(delay: int, horizon: int, prefix: int) -> None:
    batched = _policy(2, delay=delay, horizon=horizon, prefix=prefix)
    serial = _policy(2, delay=delay, horizon=horizon, prefix=prefix)
    requests = tuple(
        replace(_request(batched, stream_id, 3, 4, 0), fixed_actions=(NEUTRAL_CONTROLLER_ACTION,) * prefix)
        for stream_id in (42, 7)
    )
    actual = batched.predict(requests)
    expected = tuple(serial.predict((request,))[0] for request in reversed(requests))
    _assert_plans_match(actual, expected)
    assert all(plan.actions[0].target_frame == 4 + prefix for plan in actual)
    assert batched._rng.state() == serial._rng.state()


def test_ready_rows_keep_independent_identity_return_temperature_and_dummy_rng() -> None:
    batched = _policy(4)
    serial = _policy(4)
    settings = ((42, "PLATINUM", None, 0.8), (7, "DIAMOND", 140.0, 1.1), (9001, "MASTER", 120.0, 1.0))
    first = tuple(
        _conditioned_request(batched, stream_id, 3, 4, 0, identity=identity, desired_return=value, temperature=temp)
        for stream_id, identity, value, temp in settings
    )
    actual = batched.predict(first)
    expected = tuple(serial.predict((request,))[0] for request in reversed(first))
    _assert_plans_match(actual, expected)
    assert batched._rng.state() == serial._rng.state()
    assert len({batched._prediction_streams[stream_id].player_id for stream_id, *_ in settings}) == 3

    next_requests = tuple(
        _conditioned_request(batched, stream_id, 4, 1, 1, identity=identity, desired_return=value, temperature=temp)
        for stream_id, identity, value, temp in reversed(settings)
    )
    actual = batched.predict(next_requests)
    expected = tuple(serial.predict((request,))[0] for request in next_requests)
    _assert_plans_match(actual, expected)
    assert batched._rng.state() == serial._rng.state()
    assert len(batched._rng.state()) == 3 * len(batched._rng.index_by_group)


def test_long_observation_updates_decompose_without_dummy_time_frames() -> None:
    batched = _policy(2)
    serial = _policy(2)
    source = -1
    for sequence, count in enumerate((5, 6, 7)):
        source += count
        requests = tuple(_request(batched, stream_id, source, count, sequence) for stream_id in (42, 7))
        actual = batched.predict(requests)
        expected = tuple(serial.predict((request,))[0] for request in reversed(requests))
        _assert_plans_match(actual, expected)
        for stream_id in (42, 7):
            cache = batched._prediction_streams[stream_id].cache
            assert cache is not None and cache.next_position.item() == source + 1
        assert batched._rng.state() == serial._rng.state()


def test_sparse_ready_pair_keeps_two_row_bucket_with_32_admitted() -> None:
    sparse = _policy(32)
    pair = _policy(2)
    stream_ids = tuple(1000 + 37 * index for index in range(32))
    ready = (stream_ids[7], stream_ids[28])
    for stream_id in stream_ids:
        sparse.predict((_request(sparse, stream_id, 0, 1, 0),))
    for stream_id in ready:
        pair.predict((_request(pair, stream_id, 0, 1, 0),))
    idle_before = tuple(entry for entry in sparse._rng.state() if entry[0] not in ready)
    calls: list[int] = []
    trunk = sparse._kv_trunk

    def record_trunk(features: dict[str, torch.Tensor], observed: torch.Tensor, cache: KVCache) -> torch.Tensor:
        calls.append(observed.shape[0])
        return trunk(features, observed, cache)

    sparse._kv_trunk = record_trunk
    requests = tuple(_request(sparse, stream_id, 1, 1, 1) for stream_id in ready)
    actual = sparse.predict(requests)
    expected = tuple(pair.predict((request,))[0] for request in requests)
    assert calls == [2]
    _assert_plans_match(actual, expected)
    assert tuple(entry for entry in sparse._rng.state() if entry[0] not in ready) == idle_before


@pytest.mark.parametrize("cuda_graphs", [False, True])
def test_full_and_partial_batches_share_persistent_cache_through_wrap_and_reuse(
    cuda_graphs: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if cuda_graphs and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda" if cuda_graphs else "cpu")
    batched = _policy(4, device=device, compiled=cuda_graphs, cuda_graphs=cuda_graphs, horizon=8, prefix=3)
    serial = _policy(4, device=device, horizon=8, prefix=3)
    pool = batched._cache_pool
    assert pool is not None
    transfers: list[str] = []
    gather = pool.gather
    scatter = pool.scatter

    def record_gather(rows: tuple[int, ...], bucket: int, *, direct: bool = True) -> KVCache:
        transfers.append("gather")
        return gather(rows, bucket, direct=direct)

    def record_scatter(rows: tuple[int, ...], cache: KVCache) -> None:
        transfers.append("scatter")
        scatter(rows, cache)

    monkeypatch.setattr(pool, "gather", record_gather)
    monkeypatch.setattr(pool, "scatter", record_scatter)
    addresses = tuple(value.data_ptr() for value in pool.storage.buffers())
    frames = {stream: -1 for stream in (42, 7, 9001, 80, 123)}
    sequences = dict.fromkeys(frames, 0)
    waves = (
        ((42, 7, 9001, 80), 4),
        ((80, 9001, 7, 42), 4),
        ((7, 42, 80, 9001), 4),
        ((9001, 42), 3),
        ((80, 42, 9001, 7), 7),
        ((123, 9001, 80, 42), 2),
    )
    with count_compilation_starts() as compilation, torch.compiler.set_stance("fail_on_recompile"):
        for wave, (ready, count) in enumerate(waves):
            if wave == len(waves) - 1:
                batched.release_stream(7)
                serial.release_stream(7)
            requests = tuple(
                _conditioned_request(
                    batched,
                    stream,
                    frames[stream] + count,
                    count,
                    sequences[stream],
                    identity="PLATINUM" if stream % 2 else "MASTER",
                    desired_return=None if stream % 2 else 30.0,
                    temperature=0.8 if stream % 2 else 1.1,
                    prefix=3,
                )
                for stream in ready
            )
            before = tuple(value.clone() for value in pool.storage.buffers())
            transfers.clear()
            actual = batched.predict(requests)
            expected = tuple(serial.predict((request,))[0] for request in requests)
            assert transfers == ([] if len(ready) == 4 else ["gather", "scatter"])
            assert [plan.stream_id for plan in actual] == list(ready)
            if not cuda_graphs:
                _assert_plans_match(actual, expected)
            assert batched._rng.state() == serial._rng.state()
            assert tuple(value.data_ptr() for value in pool.storage.buffers()) == addresses
            for stream, state in batched._prediction_streams.items():
                reference = serial._prediction_streams[stream].cache
                assert state.cache is not None and reference is not None
                torch.testing.assert_close(state.cache.positions, reference.positions, rtol=0, atol=0)
                torch.testing.assert_close(state.cache.next_position, reference.next_position, rtol=0, atol=0)
                valid = state.cache.positions[0] >= 0
                tolerance = 0.035 if cuda_graphs else 2e-5
                torch.testing.assert_close(state.cache.hidden, reference.hidden, atol=tolerance, rtol=tolerance)
                for value, control in zip(
                    (*state.cache.layers, state.cache.history), (*reference.layers, reference.history), strict=True
                ):
                    torch.testing.assert_close(
                        value[:, :, :, valid], control[:, :, :, valid], atol=tolerance, rtol=tolerance
                    )
                if stream not in ready:
                    row = state.row
                    for old, new in zip(before, pool.storage.buffers(), strict=True):
                        axis = 1 if new.ndim == 5 else 0
                        torch.testing.assert_close(new.select(axis, row), old.select(axis, row), rtol=0, atol=0)
            for stream in ready:
                frames[stream] += count
                sequences[stream] += 1
        assert compilation.snapshot() == 0


def test_arbitrary_ids_release_reuse_and_capacity() -> None:
    policy = _policy(2)
    policy.predict((_request(policy, 42, 0, 1, 0), _request(policy, 7, 0, 1, 0)))
    first_row = policy._prediction_streams[42].row
    first_history = policy._prediction_streams[42].history
    first_updates = policy._prediction_streams[42].gpu
    unaffected = policy._prediction_streams[7].cache
    policy.release_stream(42)
    assert all(entry[0] != 42 for entry in policy._rng.state())
    policy.predict((_request(policy, 9001, 0, 1, 0),))
    assert policy._prediction_streams[9001].row == first_row
    assert policy._prediction_streams[9001].history is first_history
    assert policy._prediction_streams[9001].gpu is first_updates
    assert policy._prediction_streams[7].cache is unaffected
    try:
        policy.predict((_request(policy, 300, 0, 1, 0),))
    except ValueError as error:
        assert "capacity" in str(error)
    else:
        raise AssertionError("an unprepared stream was admitted")


def test_prepared_observation_layout_rejects_changed_scalar_dtypes() -> None:
    policy = _policy(1)
    request = _request(policy, 42, 0, 1, 0)
    item = request.observations[0]
    changed = replace(item, observation={**item.observation, "stage": 0.0})
    with pytest.raises(ValueError, match="noncanonical scalar type"):
        policy.predict((replace(request, observations=(changed,)),))
    assert policy._free_rows == {0}
    assert policy.predict((request,))[0].stream_id == 42


@pytest.mark.parametrize("has_history", [False, True])
@pytest.mark.parametrize("invalid_index", [0, 1])
def test_invalid_observation_rejects_complete_batch_without_advancing_streams(
    has_history: bool, invalid_index: int
) -> None:
    policy = _policy(2)
    control = _policy(2)
    if has_history:
        initial = tuple(_request(policy, stream, 1, 2, 0) for stream in (7, 42))
        policy.predict(initial)
        control.predict(initial)
    source = 3 if has_history else 1
    sequence = 1 if has_history else 0
    requests = tuple(_request(policy, stream, source, 2, sequence) for stream in (7, 42))
    observations = list(requests[1].observations)
    item = observations[invalid_index]
    observations[invalid_index] = replace(item, observation={**item.observation, "stage": 0.5})
    invalid = replace(requests[1], observations=tuple(observations))
    cursors = {
        stream_id: (stream.last_frame, stream.sequence, stream.history.written)
        for stream_id, stream in policy._prediction_streams.items()
    }
    free_rows = policy._free_rows.copy()

    with pytest.raises(ValueError, match="noncanonical scalar type"):
        policy.predict((requests[0], invalid))

    assert policy._free_rows == free_rows
    assert {
        stream_id: (stream.last_frame, stream.sequence, stream.history.written)
        for stream_id, stream in policy._prediction_streams.items()
    } == cursors
    assert policy.predict(requests) == control.predict(requests)


@pytest.mark.parametrize("port", [1, 2])
@pytest.mark.parametrize(
    "is_compiled",
    [False, pytest.param(True, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_cached_preparation_matches_live_frames_with_missing_categories(port: int, is_compiled: bool) -> None:
    device = torch.device("cuda" if is_compiled else "cpu")
    policy = _policy(1, device=device, compiled=is_compiled, cuda_graphs=is_compiled)
    game = melee.gamestate.GameState()
    game._canonical.ports = {
        player_port: melee.gamestate.PortData(
            leader=melee.gamestate.Data(post=melee.gamestate.Post(character=1, stock=4, action=14))
        )
        for player_port in (1, 2)
    }
    reference = None
    for frame_id in range(12):
        game._canonical.id = frame_id
        game._canonical.items = [melee.gamestate.Item(type=1, state=3, id=9, owner=0)] if frame_id % 3 == 1 else []
        frame = canonical_frame(game)
        item = policy_input_from_frame(
            frame,
            spec=policy.spec,
            stream_id=42,
            controlled_port=port,
            reset=frame_id == 0,
            matchup_characters={1: 1, 2: 1},
        )
        if reference is None:
            reference = ObservationHistory.from_frame(
                flatten_live_frame(frame, {1: 1, 2: 1}),
                f"p{port}",
                policy.stats,
                policy.context_frames,
                ITEM_COLUMNS,
                BASE_ITEMS_PROJECTION,
            )
            warmup = make_warmup_observations(policy.spec, 1, 42, 0, 0)[0]
            assert {name: type(value) for name, value in item.observation.items()} == {
                name: type(value) for name, value in warmup.observation.items()
            }
        reference.gather(flatten_live_frame(frame, {1: 1, 2: 1}), np.zeros(ACTION_DIM, dtype=np.float32))
        reference.push()
        with torch.compiler.set_stance("fail_on_recompile"):
            plan = policy.predict((PredictionRequest(42, 1, frame_id, frame_id, (item,), ()),))[0]
        assert len(plan.actions) == 4
        actual = policy._prediction_streams[42].history
        for name in ("values", "cats", "masks"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))


def test_cached_stream_reset_and_conditioning_keep_prepared_storage() -> None:
    policy = _policy(1)
    first = _request(policy, 42, 3, 4, 0)
    first_plan = policy.predict((first,))[0]
    original = policy._prediction_streams[42].cache
    original_history = policy._prediction_streams[42].history
    original_updates = policy._prediction_streams[42].gpu
    assert original is not None
    addresses = tuple(value.data_ptr() for value in original.buffers())
    observation_addresses = tuple(
        value.data_ptr() for value in (original_updates.floats, original_updates.cats, original_updates.actions)
    )
    for sequence, target, temperature in ((1, None, 0.8), (2, 40.0, 1.1)):
        frame = 3 + sequence
        item = make_warmup_observations(policy.spec, 1, 42, frame, 0, reset_first=False)[0]
        item = replace(item, desired_return=target, temperature=temperature)
        policy.predict((PredictionRequest(42, 1, sequence, frame, (item,), ()),))
    current = policy._prediction_streams[42].cache
    assert current is not None
    assert tuple(value.data_ptr() for value in current.buffers()) == addresses
    reset_item = make_warmup_observations(policy.spec, 1, 42, 100, 0)[0]
    policy.predict((PredictionRequest(42, 2, 0, 100, (reset_item,), ()),))
    rematch = policy._prediction_streams[42].cache
    assert rematch is not None and rematch.next_position.item() == 1
    assert tuple(value.data_ptr() for value in rematch.buffers()) == addresses
    assert policy._prediction_streams[42].history is original_history
    assert policy._prediction_streams[42].gpu is original_updates
    assert (
        tuple(value.data_ptr() for value in (original_updates.floats, original_updates.cats, original_updates.actions))
        == observation_addresses
    )
    policy.reset_prediction(seed=5)
    assert policy.predict((first,))[0] == first_plan

    changed = replace(first.observations[-1], frame_id=4, reset=False, player_identity="PLATINUM")
    with pytest.raises(ValueError, match="identity changed"):
        policy.predict((PredictionRequest(42, 1, 1, 4, (changed,), ()),))


def test_artifact_window_mode_delegates_to_canonical_dense_executor() -> None:
    policy = _policy(2, history_mode="window")
    assert policy._dense_policy is not None
    assert policy._cache_pool is None
    requests = (_request(policy, 42, 3, 4, 0), _request(policy, 7, 3, 4, 0))
    plans = policy.predict(requests)
    assert len(plans) == 2
    assert [plan.actions[0].target_frame for plan in plans] == [4, 4]
    policy.reset_prediction(seed=19)
    assert policy.sampling_seed == 19

    policy.prepare_prediction(RuntimeConfig(2, (2,)), 4, 2)
    prefix = (NEUTRAL_CONTROLLER_ACTION,) * 2
    request = replace(_request(policy, 9001, 3, 4, 0), fixed_actions=prefix)
    plan = policy.predict((request,))[0]
    assert len(plan.actions) == 2
    assert plan.actions[0].target_frame == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_window_buckets_are_captured_before_sparse_admission() -> None:
    policy = _policy(4, device=torch.device("cuda"), compiled=True, history_mode="window")
    assert policy._dense_policy is not None
    executor = policy._dense_policy.executor
    assert {bucket for bucket, _, _ in executor._warmed} == {1, 2, 4}
    captured = set(executor._warmed)
    with torch.compiler.set_stance("fail_on_recompile"):
        for count in (1, 2, 3, 4):
            requests = tuple(_request(policy, count * 100 + row, 3, 4, 0) for row in range(count))
            assert len(policy.predict(requests)) == count
            for request in requests:
                policy.release_stream(request.stream_id)
    assert executor._warmed == captured


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("capacity", [1, 2])
def test_cuda_graphs_are_captured_before_stream_admission(capacity: int) -> None:
    policy = _policy(capacity, device=torch.device("cuda"), compiled=True, cuda_graphs=True)
    buckets = (1,) if capacity == 1 else (1, 2)
    assert set(policy._update_calls) == {(bucket, count) for bucket in buckets for count in (1, 2, 4)}
    assert set(policy._decoder_calls) == set(buckets)
    before = (tuple(policy._update_calls), tuple(policy._decoder_calls))
    row_updates = policy._row_updates
    row_storage = tuple(
        (stage.floats.data_ptr(), stage.cats.data_ptr(), stage.actions.data_ptr(), stage.player.data_ptr())
        for stage in row_updates
    )
    for request in (
        _request(policy, 42, 3, 4, 0),
        _request(policy, 42, 4, 1, 1),
        _request(policy, 42, 7, 3, 2),
    ):
        plan = policy.predict((request,))[0]
        assert len(plan.actions) == 4
    if capacity == 2:
        old_row = policy._prediction_streams[42].row
        policy.release_stream(42)
        plans = policy.predict((_request(policy, 9001, 3, 4, 0), _request(policy, 7, 3, 4, 0)))
        assert len(plans) == 2
        assert policy._prediction_streams[9001].row == old_row
        assert policy._prediction_streams[9001].gpu is row_updates[old_row]
        reset_item = make_warmup_observations(policy.spec, 1, 9001, 100, 0)[0]
        policy.predict((PredictionRequest(9001, 2, 0, 100, (reset_item,), ()),))
        assert policy._prediction_streams[9001].gpu is row_updates[old_row]
    assert (
        tuple(
            (stage.floats.data_ptr(), stage.cats.data_ptr(), stage.actions.data_ptr(), stage.player.data_ptr())
            for stage in policy._row_updates
        )
        == row_storage
    )
    assert (tuple(policy._update_calls), tuple(policy._decoder_calls)) == before


def _exercise_shared_profile_interleave(
    device: torch.device, *, capacity: int, compiled: bool, cuda_graphs: bool
) -> None:
    first = _policy(capacity, device=device, compiled=compiled, cuda_graphs=cuda_graphs)
    second = _policy(
        capacity,
        model=first.model,
        device=device,
        compiled=compiled,
        cuda_graphs=cuda_graphs,
        delay=2,
        horizon=8,
        prefix=3,
    )
    third = _policy(
        capacity,
        model=first.model,
        device=device,
        compiled=compiled,
        cuda_graphs=cuda_graphs,
        delay=3,
        horizon=8,
        prefix=4,
    )
    policies = (first, second, third)
    assert first.model is second.model is third.model
    prepared = tuple((tuple(policy._update_calls), tuple(policy._decoder_calls)) for policy in policies)
    addresses = tuple(
        policy._cache_pool.storage.hidden.data_ptr() for policy in policies if policy._cache_pool is not None
    )
    assert len(set(addresses)) == 3
    requests = (
        _request(first, 42, 3, 4, 0),
        replace(_request(second, 7, 3, 4, 0), fixed_actions=(NEUTRAL_CONTROLLER_ACTION,) * 3),
        replace(_request(third, 9001, 3, 4, 0), fixed_actions=(NEUTRAL_CONTROLLER_ACTION,) * 4),
    )
    if compiled:
        # Qualification prepares another profile after resetting Dynamo's cache.
        # Earlier CUDA graphs must still execute without any new compilation.
        torch.compiler.reset()
    with torch.compiler.set_stance("fail_on_recompile"):
        a = first.predict((requests[0],))[0]
        first_cache = first._prediction_streams[42].cache
        assert first_cache is not None
        first_position = first_cache.next_position.clone()
        b = second.predict((requests[1],))[0]
        c = third.predict((requests[2],))[0]
        torch.testing.assert_close(first_cache.next_position, first_position, rtol=0, atol=0)
        a_next = first.predict((_request(first, 42, 4, 1, 1),))[0]
    assert tuple(plan.actions[0].target_frame for plan in (a, b, c, a_next)) == (4, 7, 8, 5)
    assert tuple(policy.prediction_horizon for policy in policies) == (4, 8, 8)
    assert tuple(policy._prefix_frames for policy in policies) == (0, 3, 4)
    assert tuple((tuple(policy._update_calls), tuple(policy._decoder_calls)) for policy in policies) == prepared
    assert all(policy._prediction_streams for policy in policies)


def test_shared_model_keeps_distinct_cached_profiles_on_cpu() -> None:
    _exercise_shared_profile_interleave(torch.device("cpu"), capacity=1, compiled=False, cuda_graphs=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_model_keeps_captured_cached_profiles_without_recompile() -> None:
    _exercise_shared_profile_interleave(torch.device("cuda"), capacity=2, compiled=True, cuda_graphs=True)


@pytest.mark.integration
@torch.inference_mode()
def test_cuda_graph_cache_matches_eager_through_wrap_reset_and_settings() -> None:
    if os.environ.get("HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 on the production GPU")
    if not torch.cuda.is_available():
        pytest.fail("CUDA is required for cached graph qualification")
    device = torch.device("cuda")
    compiled = _policy(1, device=device, compiled=True, cuda_graphs=True, delay=2, horizon=8, prefix=3)
    eager = _policy(1, device=device, compiled=False, cuda_graphs=False, delay=2, horizon=8, prefix=3)
    prefix = (NEUTRAL_CONTROLLER_ACTION,) * 3
    addresses: tuple[int, ...] | None = None
    with torch.compiler.set_stance("fail_on_recompile"):
        for frame in range(45):
            item = make_warmup_observations(compiled.spec, 1, 0, frame, 2)[0]
            item = replace(
                item,
                reset=frame in (0, 24),
                observation={**item.observation, "p1_percent": float(frame)},
                desired_return=None if frame % 3 == 0 else 20.0,
                temperature=0.8 if frame % 2 else 1.1,
            )
            request = PredictionRequest(0, 1 if frame < 24 else 2, frame, frame, (item,), prefix)
            compiled.predict((request,))
            eager.predict((request,))
            cache = compiled._prediction_streams[0].cache
            reference = eager._prediction_streams[0].cache
            assert cache is not None and reference is not None
            current = tuple(value.data_ptr() for value in cache.buffers())
            if addresses is None:
                addresses = current
            assert current == addresses
            torch.testing.assert_close(cache.next_position, reference.next_position, rtol=0, atol=0)
            torch.testing.assert_close(cache.positions, reference.positions, rtol=0, atol=0)
            torch.testing.assert_close(cache.hidden, reference.hidden, atol=0.035, rtol=0.035)
            valid = cache.positions[0] >= 0
            for actual, expected in zip(cache.layers, reference.layers, strict=True):
                torch.testing.assert_close(actual[:, :, :, valid], expected[:, :, :, valid], atol=0.035, rtol=0.035)
            torch.testing.assert_close(
                cache.history[:, :, :, valid], reference.history[:, :, :, valid], atol=0.035, rtol=0.035
            )


@pytest.mark.parametrize("history_mode", ["window", "kv_cache"])
def test_prediction_returns_latest_value_without_changing_actions_or_rng(history_mode: str) -> None:
    policy = _policy(2, history_mode=history_mode)
    control = _policy(2, history_mode=history_mode)
    requests = tuple(_request(policy, stream_id, 7, 8, 0) for stream_id in (7, 42))
    seen: list[torch.Tensor] = []
    estimate = policy.model.estimate_value

    def record(hidden: torch.Tensor) -> torch.Tensor:
        seen.append(hidden.detach().clone())
        return estimate(hidden)

    policy.model.estimate_value = record
    actual = policy.predict(requests)
    expected = control.predict(requests)
    assert [p.actions for p in actual] == [p.actions for p in expected]
    assert policy._rng.state() == control._rng.state()
    assert len(seen) == 1
    with torch.no_grad():
        reference = policy.model.value_head(
            torch.nn.functional.rms_norm(seen[0][:, -1], (policy.cfg.d_model,), eps=1e-6).float()
        ).squeeze(-1)
    assert [p.state_value for p in actual] == pytest.approx(reference.tolist(), abs=2e-6, rel=2e-5)


def test_value_and_actions_follow_the_controlled_port() -> None:
    first, second = _policy(1), _policy(1)
    request = _request(first, 0, 7, 8, 0)
    observations = tuple(
        replace(item, observation=dict(item.observation) | {"p1_position_x": 10.0, "p2_position_x": -40.0})
        for item in request.observations
    )
    swapped = tuple(
        replace(
            item,
            controlled_port=2,
            observation={
                (
                    "p2_" + key[3:] if key.startswith("p1_") else "p1_" + key[3:] if key.startswith("p2_") else key
                ): value
                for key, value in item.observation.items()
            },
        )
        for item in observations
    )
    actual = first.predict((replace(request, observations=observations),))
    expected = second.predict((replace(request, observations=swapped),))
    _assert_plans_match(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_value_head_uses_serving_precision_and_returns_fp32(dtype: torch.dtype) -> None:
    policy = _policy(1)
    model = policy.model
    model.value_head.to(dtype=dtype)
    hidden = torch.randn(2, 3, model.cfg.d_model).to(dtype)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        expected = (
            model.value_head(torch.nn.functional.rms_norm(hidden[:, -1], (model.cfg.d_model,), eps=1e-6))
            .squeeze(-1)
            .float()
        )
        actual = model.estimate_value(hidden)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
