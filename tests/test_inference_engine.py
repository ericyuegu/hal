"""Weight registry identity and lazy construction."""

import threading
import time
from dataclasses import replace
from multiprocessing import Pipe
from types import SimpleNamespace

import pytest
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import action_plan
from hal.inference.client import StreamAck
from hal.inference.client import StreamAdmission
from hal.inference.client import StreamRelease
from hal.inference.engine import InferenceEngine
from hal.inference.engine import ModelRegistry
from hal.inference.engine import start_inference_worker


def test_local_worker_thread_preserves_engine_failure_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    class Policy:
        spec = PolicySpec("test", "test", (), (0,))
        context_frames = 8

        def validate_prepared_profile(self, _profile: PreparedInferenceProfile) -> None:
            pass

    def fail(_engine: InferenceEngine, _stop: object) -> None:
        raise ValueError("GPU execution failed")

    monkeypatch.setattr(InferenceEngine, "serve", fail)
    profile = PreparedInferenceProfile("single", "a" * 64, "window", 4, 0, (1,), 1)
    with (
        pytest.raises(RuntimeError, match="local inference worker failed") as raised,
        start_inference_worker(Policy(), profile, 0),
    ):
        time.sleep(0.01)
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "GPU execution failed"


def test_engine_rejects_profile_that_bound_policy_did_not_prepare() -> None:
    class Policy:
        def validate_prepared_profile(self, _profile: PreparedInferenceProfile) -> None:
            raise ValueError("prepared shape differs from policy")

    profile = PreparedInferenceProfile("test", "a" * 64, "window", 4, 2, (1, 2, 4), 1)
    parent, child = Pipe()
    try:
        with pytest.raises(ValueError, match="prepared shape differs"):
            InferenceEngine({profile: Policy()}, {0: parent}, batch_wait_seconds=0)
    finally:
        parent.close()
        child.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", True),
        ("checkpoint_sha256", True),
        ("execution_mode", True),
        ("prediction_horizon_frames", True),
        ("fixed_prefix_frames", True),
        ("capacity", True),
        ("update_shapes", (True, 2)),
        ("update_shapes", [1, 2]),
    ],
)
def test_prepared_profile_rejects_noncanonical_geometry(field: str, value: object) -> None:
    profile = PreparedInferenceProfile("test", "a" * 64, "window", 4, 0, (1, 2), 1)
    with pytest.raises(ValueError):
        replace(profile, **{field: value})


def test_registry_builds_once_per_checkpoint_device_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[object, torch.device, torch.dtype]] = []

    def build(artifact: object, *, device: torch.device, inference_dtype: torch.dtype) -> object:
        built.append((artifact, device, inference_dtype))
        return object()

    monkeypatch.setattr("hal.inference.engine.build_action_sequence_model", build)
    first = SimpleNamespace(checkpoint_sha256="sha-a", bundle_name="first")
    same_checkpoint = SimpleNamespace(checkpoint_sha256="sha-a", bundle_name="second")
    other_checkpoint = SimpleNamespace(checkpoint_sha256="sha-b", bundle_name="third")
    registry = ModelRegistry()

    model = registry.register_artifact(first, device="cpu", inference_dtype=torch.float32)
    assert registry.register_artifact(same_checkpoint, device="cpu:0", inference_dtype=torch.float32) is model
    assert registry.register_artifact(other_checkpoint, device="cpu", inference_dtype=torch.float32) is not model
    assert registry.model_count == len(built) == 2
    assert all(device == torch.device("cpu") and dtype == torch.float32 for _, device, dtype in built)

    with pytest.raises(ValueError, match="unsupported inference device/dtype"):
        registry.register_artifact(first, device="cpu", inference_dtype=torch.float16)
    with pytest.raises(ValueError, match="unsupported inference device/dtype"):
        registry.register_artifact(first, device="cpu", inference_dtype=torch.bfloat16)
    assert len(built) == 2


def test_engine_batches_only_ready_requests_with_the_same_prepared_prefix() -> None:
    call_order: list[int] = []
    early_delivery = []

    class Policy:
        def __init__(self, prefix: int) -> None:
            self.prefix = prefix
            self.batches: list[tuple[int, ...]] = []

        def predict(self, requests: tuple[PredictionRequest, ...]):
            call_order.append(self.prefix)
            if self.prefix == 3:
                early_delivery.append(pairs[2][1].poll())
            self.batches.append(tuple(request.stream_id for request in requests))
            return tuple(
                action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * (8 - len(request.fixed_actions)))
                for request in requests
            )

        def release_stream(self, _stream_id: int) -> None:
            pass

        def validate_prepared_profile(self, profile: PreparedInferenceProfile) -> None:
            assert profile.fixed_prefix_frames == self.prefix

    prefix_three = Policy(3)
    prefix_four = Policy(4)
    pairs = [Pipe() for _ in range(4)]
    try:
        profile_three = PreparedInferenceProfile("delay-2", "a" * 64, "kv_cache", 8, 3, (1, 2, 4), 2)
        profile_four = PreparedInferenceProfile("delay-3", "a" * 64, "kv_cache", 8, 4, (1, 2, 4), 2)
        engine = InferenceEngine(
            {profile_three: prefix_three, profile_four: prefix_four},
            {index: parent for index, (parent, _) in enumerate(pairs)},
            batch_wait_seconds=0.0005,
        )
        for index, (stream_id, profile) in enumerate(((17, profile_three), (98, profile_three), (400, profile_four))):
            pairs[index][1].send(StreamAdmission(stream_id, 1, profile))
        engine.serve_batch([pairs[index][0] for index in range(3)], 0.0)
        for index, stream_id in enumerate((17, 98, 400)):
            assert pairs[index][1].recv() == StreamAck(stream_id, 1, "admitted")
        for stream_id, prefix in ((17, 3), (98, 3), (400, 4)):
            item = PolicyInput(stream_id, 0, 1, {}, NEUTRAL_CONTROLLER_ACTION)
            request = PredictionRequest(
                stream_id,
                1,
                0,
                0,
                (item,),
                (NEUTRAL_CONTROLLER_ACTION,) * prefix,
                10.0 if prefix == 4 else 20.0,
            )
            pairs[(17, 98, 400).index(stream_id)][1].send(request)
        engine.serve_batch([pairs[index][0] for index in range(3)], 0.0)
        assert prefix_three.batches == [(17, 98)]
        assert prefix_four.batches == [(400,)]
        assert call_order == [4, 3]
        assert early_delivery == [True]
        assert all(pairs[index][1].recv().stream_id == stream_id for index, stream_id in enumerate((17, 98, 400)))
        assert not pairs[3][1].poll()
    finally:
        for parent, child in pairs:
            parent.close()
            child.close()


def test_engine_releases_rows_across_rematches_and_rejects_other_connections() -> None:
    class Policy:
        def __init__(self) -> None:
            self.released: list[int] = []

        def release_stream(self, stream_id: int) -> None:
            self.released.append(stream_id)

        def validate_prepared_profile(self, profile: PreparedInferenceProfile) -> None:
            assert profile.capacity == 1

    profile = PreparedInferenceProfile("single", "b" * 64, "kv_cache", 8, 3, (1, 2, 4), 1)
    policy = Policy()
    first_parent, first_child = Pipe()
    second_parent, second_child = Pipe()
    try:
        engine = InferenceEngine({profile: policy}, {0: first_parent, 1: second_parent}, batch_wait_seconds=0)
        first_child.send(StreamAdmission(8127, 1, profile))
        engine.serve_batch((first_parent,), 0)
        assert first_child.recv() == StreamAck(8127, 1, "admitted")

        first_child.send(StreamAdmission(8127, 2, profile))
        engine.serve_batch((first_parent,), 0)
        assert first_child.recv() == StreamAck(8127, 2, "admitted")
        assert policy.released == [8127]

        first_child.send(StreamRelease(8127, 2))
        engine.serve_batch((first_parent,), 0)
        assert first_child.recv() == StreamAck(8127, 2, "released")
        assert policy.released == [8127, 8127]

        second_child.send(StreamAdmission(999_999, 1, profile))
        engine.serve_batch((second_parent,), 0)
        assert second_child.recv() == StreamAck(999_999, 1, "admitted")
        first_child.send(StreamAdmission(999_999, 2, profile))
        with pytest.raises(ValueError, match="connection"):
            engine.serve_batch((first_parent,), 0)
    finally:
        first_parent.close()
        first_child.close()
        second_parent.close()
        second_child.close()


def test_sparse_ready_pair_does_not_wait_for_idle_admissions(monkeypatch: pytest.MonkeyPatch) -> None:
    import hal.inference.engine as runtime

    stop = threading.Event()

    class Policy:
        def predict(self, requests: tuple[PredictionRequest, ...]):
            assert len(requests) == 2
            stop.set()
            return tuple(action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 5) for request in requests)

        def release_stream(self, _stream_id: int) -> None:
            pass

        def validate_prepared_profile(self, profile: PreparedInferenceProfile) -> None:
            assert profile.capacity == 32

    profile = PreparedInferenceProfile("sparse", "c" * 64, "kv_cache", 8, 3, (1, 2, 4), 32)
    pairs = [Pipe() for _ in range(32)]
    try:
        engine = InferenceEngine(
            {profile: Policy()}, {i: parent for i, (parent, _) in enumerate(pairs)}, batch_wait_seconds=0.0005
        )
        for index, (_, child) in enumerate(pairs):
            child.send(StreamAdmission(1000 + index, 1, profile))
        engine.serve_batch(tuple(parent for parent, _ in pairs), 0)
        for index, (_, child) in enumerate(pairs):
            assert child.recv() == StreamAck(1000 + index, 1, "admitted")
        for index in (0, 1):
            stream_id = 1000 + index
            item = PolicyInput(stream_id, 0, 1, {}, NEUTRAL_CONTROLLER_ACTION)
            pairs[index][1].send(PredictionRequest(stream_id, 1, 0, 0, (item,), (NEUTRAL_CONTROLLER_ACTION,) * 3))
        waits: list[float | None] = []
        original_wait = runtime.wait

        def recorded_wait(connections, timeout=None):
            waits.append(timeout)
            return original_wait(connections, timeout=timeout)

        monkeypatch.setattr(runtime, "wait", recorded_wait)
        engine.serve(stop)
        assert engine.batch_items == 2 and engine.batch_calls == 1
        assert all(timeout in (0, 0.05) for timeout in waits)
        assert pairs[0][1].recv().stream_id == 1000
        assert pairs[1][1].recv().stream_id == 1001
    finally:
        for parent, child in pairs:
            parent.close()
            child.close()
