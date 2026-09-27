"""Prepared dense buckets cover every admitted ready-batch size."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hal.inference.api import PolicySpec
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.window_policy import DenseWindowPredictionPolicy


class _Executor:
    compiled = False
    context_frames = 8
    prediction_frames = 4
    prepared_buckets = (1, 2, 4, 8)
    model = SimpleNamespace(head_offsets=(1, 2, 3, 4))

    def __init__(self) -> None:
        self.warmed: list[tuple[int, int, int]] = []

    def _bucket(self, rows: int) -> int:
        return next(bucket for bucket in self.prepared_buckets if bucket >= rows)

    def prewarm(self, rows: int, horizon: int, *, committed_frames: int) -> float:
        self.warmed.append((rows, horizon, committed_frames))
        return 0.0


def test_dense_preparation_warms_all_covering_buckets() -> None:
    executor = _Executor()
    policy = DenseWindowPredictionPolicy(
        executor,
        {},
        seed=0,
        spec=PolicySpec("test", "test", (), (0,)),
        checkpoint_sha256="a" * 64,
    )
    policy.prepare_prediction(RuntimeConfig(5, (0,)), 4, 2)
    assert executor.warmed == [(1, 4, 2), (2, 4, 2), (4, 4, 2), (8, 4, 2)]
    profile = PreparedInferenceProfile("dense", "a" * 64, "window", 4, 2, (1, 2, 4), 5)
    policy.validate_prepared_profile(profile)
    with pytest.raises(ValueError, match="differs"):
        policy.validate_prepared_profile(replace(profile, fixed_prefix_frames=1))


def test_failed_dense_warmup_cannot_be_admitted() -> None:
    executor = _Executor()
    executor.compiled = True
    policy = DenseWindowPredictionPolicy(executor, {}, seed=0, checkpoint_sha256="a" * 64)

    def fail(_rows: int, _horizon: int, *, committed_frames: int) -> float:
        assert committed_frames == 2
        raise RuntimeError("capture failed")

    executor.prewarm = fail
    with pytest.raises(RuntimeError, match="capture failed"):
        policy.prepare_prediction(RuntimeConfig(2, (0,)), 4, 2)
    profile = PreparedInferenceProfile("dense", "a" * 64, "window", 4, 2, (1, 2), 2)
    with pytest.raises(ValueError, match="lacks prepared"):
        policy.validate_prepared_profile(profile)
