"""Check calibration selection against exhaustive, non-monotone timings."""

import random
from typing import cast
from unittest.mock import Mock

import pytest

from hal.eval.qualification import check_realtime_budget
from hal.eval.qualification import select_frame_timing
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PredictionPolicy
from hal.inference.api import RuntimeConfig
from hal.inference.benchmark import LatencyMeasurement


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("transport", (2, 3))
def test_early_selection_matches_exhaustive_nonmonotone_timings(seed: int, transport: int) -> None:
    rng = random.Random(seed)
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = tuple(range(1, 13))
    samples = {}
    for horizon in policy.supported_horizons:
        for budget in range(1, (horizon - transport) // 2 + 1):
            prefix = transport + budget
            samples[horizon, prefix] = LatencyMeasurement(
                horizon, prefix, (rng.choice((0.001, 0.018, 0.036, 0.058, 0.080)),)
            )
    calls: list[tuple[int, int]] = []

    def measure(
        policy: PredictionPolicy, runtime: RuntimeConfig, horizon: int, prefix: int, batch_wait: float
    ) -> LatencyMeasurement:
        calls.append((horizon, prefix))
        return samples[horizon, prefix]

    runtime = RuntimeConfig(1, (transport,))
    try:
        expected = select_frame_timing(tuple(samples.values()), transport)
    except RuntimeError:
        with pytest.raises(RuntimeError, match="no trained horizon"):
            check_realtime_budget(cast(PredictionPolicy, policy), runtime, 0.0005, measure=measure)
        return
    result = check_realtime_budget(cast(PredictionPolicy, policy), runtime, 0.0005, measure=measure)
    assert result.timings == (expected,)
    assert calls[-2:] == [(expected.prediction_horizon_frames, expected.fixed_prefix_frames)] * 2
    assert len(calls) <= len(samples) + 1
    policy.reset_prediction.assert_called_once()


def test_best_candidate_needs_only_two_measurements() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = tuple(range(1, 13))
    measure = Mock(return_value=LatencyMeasurement(12, 3, (0.010,)))
    result = check_realtime_budget(cast(PredictionPolicy, policy), RuntimeConfig(1, (2,)), 0.0005, measure=measure)
    assert result.timings == (FrameTiming(2, 1, 3, 1, 12),)
    assert measure.call_count == 2
    assert all(call.args[2:4] == (12, 3) for call in measure.call_args_list)


def test_replan_interval_can_be_set_independently_of_handoff() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (5,)
    measure = Mock(return_value=LatencyMeasurement(5, 3, (0.010,)))
    runtime = RuntimeConfig(1, (2,), replan_interval_frames=1)
    result = check_realtime_budget(cast(PredictionPolicy, policy), runtime, 0.0005, measure=measure)
    assert result.timings == (FrameTiming(2, 1, 3, 1, 5),)


def test_one_frame_thinking_allowance_accepts_one_frame_latency() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (8,)
    runtime = RuntimeConfig(1, (2,), replan_interval_frames=4)
    measure = Mock(return_value=LatencyMeasurement(8, 3, (1 / 60,)))
    result = check_realtime_budget(cast(PredictionPolicy, policy), runtime, 0.0005, measure=measure)
    assert result.timings == (FrameTiming(2, 1, 3, 4, 8),)
    assert result.timings[0].reserve_frames == 1
    assert measure.call_count == 2


def test_one_frame_thinking_allowance_rejects_later_result() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (8,)
    runtime = RuntimeConfig(1, (2,), replan_interval_frames=4)
    measure = Mock(return_value=LatencyMeasurement(8, 3, (1 / 60 + 1e-9,)))
    with pytest.raises(RuntimeError, match="failed final qualification"):
        check_realtime_budget(cast(PredictionPolicy, policy), runtime, 0.0005, shape=(8, 3), measure=measure)


def test_selected_shape_must_pass_independent_qualification() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (12,)
    measure = Mock(side_effect=(LatencyMeasurement(12, 3, (0.010,)), LatencyMeasurement(12, 3, (0.020,))))
    with pytest.raises(RuntimeError, match="failed final qualification") as failure:
        check_realtime_budget(cast(PredictionPolicy, policy), RuntimeConfig(1, (2,)), 0.0005, measure=measure)
    assert "delay=2 horizon=12 fixed_prefix=3 p99=20.000ms allowance=16.667ms samples=1" in str(failure.value)


def test_second_delay_failure_reports_measured_profile() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (8,)
    measure = Mock(side_effect=(LatencyMeasurement(8, 4, (0.010,)), LatencyMeasurement(8, 3, (0.020,))))
    runtime = RuntimeConfig(1, (2, 3), replan_interval_frames=4)
    with pytest.raises(RuntimeError, match="failed independent qualification") as failure:
        check_realtime_budget(cast(PredictionPolicy, policy), runtime, 0.0005, shape=(8, 4), measure=measure)
    assert "delay=2 horizon=8 fixed_prefix=3 p99=20.000ms allowance=16.667ms samples=1" in str(failure.value)


def test_unusable_horizons_fail_without_compiling() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (1, 2, 3)
    measure = Mock()
    with pytest.raises(RuntimeError, match="no trained horizon"):
        check_realtime_budget(cast(PredictionPolicy, policy), RuntimeConfig(1, (2,)), 0.0005, measure=measure)
    measure.assert_not_called()


def test_explicit_debug_shape_is_qualified_without_search() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (12,)
    measure = Mock(return_value=LatencyMeasurement(12, 6, (0.035,)))
    result = check_realtime_budget(
        cast(PredictionPolicy, policy), RuntimeConfig(1, (2,)), 0.0005, shape=(12, 6), measure=measure
    )
    assert result.timings == (FrameTiming(2, 4, 6, 4, 12),)
    assert measure.call_count == 1


def test_delay_two_and_three_have_distinct_measured_prefixes() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (8,)
    calls: list[tuple[int, int]] = []

    def measure(
        _policy: PredictionPolicy, _runtime: RuntimeConfig, horizon: int, prefix: int, _wait: float
    ) -> LatencyMeasurement:
        calls.append((horizon, prefix))
        return LatencyMeasurement(horizon, prefix, (0.010,))

    result = check_realtime_budget(
        cast(PredictionPolicy, policy),
        RuntimeConfig(2, (2, 3), replan_interval_frames=4),
        0.0005,
        shape=(8, 4),
        measure=measure,
    )
    assert result.timings == (FrameTiming(2, 1, 3, 4, 8), FrameTiming(3, 1, 4, 4, 8))
    assert calls == [(8, 4), (8, 3)]


def test_measurement_shape_must_match_requested_shape() -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (12,)
    measure = Mock(return_value=LatencyMeasurement(12, 5, (0.010,)))
    with pytest.raises(ValueError, match="measurement shape"):
        check_realtime_budget(
            cast(PredictionPolicy, policy), RuntimeConfig(1, (2,)), 0.0005, shape=(12, 6), measure=measure
        )


@pytest.mark.parametrize("seconds", (4 / 60 + 1e-9, 0.1))
def test_explicit_debug_shape_still_rejects_excess_latency(seconds: float) -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (12,)
    measure = Mock(return_value=LatencyMeasurement(12, 6, (seconds,)))
    with pytest.raises(RuntimeError, match="failed final qualification"):
        check_realtime_budget(
            cast(PredictionPolicy, policy), RuntimeConfig(1, (2,)), 0.0005, shape=(12, 6), measure=measure
        )


@pytest.mark.parametrize("shape", ((13, 6), (12, 2), (12, 9)))
def test_explicit_debug_shape_rejects_unsupported_or_invalid_schedule(shape: tuple[int, int]) -> None:
    policy = Mock(spec=PredictionPolicy)
    policy.supported_horizons = (12,)
    measure = Mock()
    with pytest.raises(ValueError):
        check_realtime_budget(
            cast(PredictionPolicy, policy), RuntimeConfig(1, (2,)), 0.0005, shape=shape, measure=measure
        )
    measure.assert_not_called()
