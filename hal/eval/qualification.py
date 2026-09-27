"""Qualify prediction timing before accepting a live match."""

import math
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

from hal.eval.scheduling import FrameTiming
from hal.inference.api import PredictionPolicy
from hal.inference.api import RuntimeConfig
from hal.inference.benchmark import LatencyMeasurement
from hal.inference.benchmark import measure_prediction_shape


@dataclass(frozen=True, slots=True)
class RealtimeBudgetCheck:
    timings: tuple[FrameTiming, ...]
    measurements: tuple[LatencyMeasurement, ...]


def _timing_order(timing: FrameTiming) -> tuple[int, int]:
    return timing.inference_allowance_frames, -timing.prediction_horizon_frames


def _candidate_order(candidate: tuple[int, int]) -> tuple[int, int]:
    return candidate[0], -candidate[1]


def latency_frames(seconds: float) -> int:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("latency must be finite and non-negative")
    return math.ceil(seconds * 60)


def select_frame_timing(
    measurements: tuple[LatencyMeasurement, ...],
    input_delay_frames: int,
    replan_interval_frames: int | None = None,
) -> FrameTiming:
    candidates = []
    for measurement in measurements:
        thinking_allowance = measurement.fixed_prefix_frames - input_delay_frames
        replan = thinking_allowance if replan_interval_frames is None else replan_interval_frames
        if latency_frames(measurement.p99_seconds) <= thinking_allowance:
            try:
                timing = FrameTiming(
                    input_delay_frames,
                    thinking_allowance,
                    measurement.fixed_prefix_frames,
                    replan,
                    measurement.prediction_horizon_frames,
                )
            except ValueError:
                continue
            candidates.append(timing)
    if not candidates:
        raise RuntimeError("server unavailable: no trained horizon can cover measured inference and transport")
    return min(candidates, key=_timing_order)


def check_realtime_budget(
    policy: PredictionPolicy,
    runtime: RuntimeConfig,
    batch_wait_seconds: float,
    *,
    shape: tuple[int, int] | None = None,
    measure: Callable[
        [PredictionPolicy, RuntimeConfig, int, int, float], LatencyMeasurement
    ] = measure_prediction_shape,
) -> RealtimeBudgetCheck:
    """Select a feasible workload, then independently time the selected shape."""
    input_delay = max(runtime.transport_delays)
    candidates = []
    for horizon in policy.supported_horizons:
        for thinking_allowance in range(1, horizon - input_delay + 1):
            replan = thinking_allowance if runtime.replan_interval_frames is None else runtime.replan_interval_frames
            if horizon >= input_delay + thinking_allowance + replan:
                candidates.append((thinking_allowance, horizon))
    candidates.sort(key=_candidate_order)
    measurements = []
    if shape is not None:
        horizon, prefix = shape
        if horizon not in policy.supported_horizons:
            raise ValueError("prediction horizon is not supported by this policy")
        thinking_allowance = prefix - input_delay
        replan = thinking_allowance if runtime.replan_interval_frames is None else runtime.replan_interval_frames
        selected = FrameTiming(input_delay, thinking_allowance, prefix, replan, horizon)
    else:
        for thinking_allowance, horizon in candidates:
            logger.info("measuring prediction horizon={} fixed_prefix={}", horizon, input_delay + thinking_allowance)
            result = measure(policy, runtime, horizon, input_delay + thinking_allowance, batch_wait_seconds)
            if (result.prediction_horizon_frames, result.fixed_prefix_frames) != (
                horizon,
                input_delay + thinking_allowance,
            ):
                raise ValueError("latency measurement shape differs from the requested prediction shape")
            measurements.append(result)
            if latency_frames(result.p99_seconds) <= thinking_allowance:
                selected = FrameTiming(
                    input_delay,
                    thinking_allowance,
                    input_delay + thinking_allowance,
                    thinking_allowance if runtime.replan_interval_frames is None else runtime.replan_interval_frames,
                    horizon,
                )
                break
        else:
            raise RuntimeError("server unavailable: no trained horizon can cover measured inference and transport")
    logger.info(
        "qualifying prediction horizon={} fixed_prefix={}",
        selected.prediction_horizon_frames,
        selected.fixed_prefix_frames,
    )
    result = measure(
        policy,
        runtime,
        selected.prediction_horizon_frames,
        selected.fixed_prefix_frames,
        batch_wait_seconds,
    )
    if (result.prediction_horizon_frames, result.fixed_prefix_frames) != (
        selected.prediction_horizon_frames,
        selected.fixed_prefix_frames,
    ):
        raise ValueError("latency measurement shape differs from the requested prediction shape")
    measurements.append(result)
    if latency_frames(result.p99_seconds) > selected.inference_allowance_frames:
        raise RuntimeError("server unavailable: selected schedule failed final qualification")
    timings = tuple(
        FrameTiming(
            delay,
            selected.inference_allowance_frames,
            delay + selected.inference_allowance_frames,
            selected.replan_interval_frames,
            selected.prediction_horizon_frames,
        )
        for delay in runtime.transport_delays
    )
    for timing in timings:
        if timing.physical_delay_frames == selected.physical_delay_frames:
            continue
        result = measure(
            policy,
            runtime,
            timing.prediction_horizon_frames,
            timing.fixed_prefix_frames,
            batch_wait_seconds,
        )
        if (result.prediction_horizon_frames, result.fixed_prefix_frames) != (
            timing.prediction_horizon_frames,
            timing.fixed_prefix_frames,
        ):
            raise ValueError("latency measurement shape differs from the requested prediction shape")
        measurements.append(result)
        if latency_frames(result.p99_seconds) > timing.inference_allowance_frames:
            raise RuntimeError(
                f"server unavailable: delay {timing.physical_delay_frames} failed independent qualification"
            )
    policy.reset_prediction()
    return RealtimeBudgetCheck(timings, tuple(measurements))
