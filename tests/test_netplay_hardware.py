"""Opt-in qualification of the prepared 059 netplay profiles."""

import math
import os
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import PredictionRequest
from hal.inference.api import validate_action_plan
from hal.inference.engine import InferenceEngine
from hal.inference.engine import configure_inference_process
from hal.inference.warmup import make_warmup_observations
from hal.netplay_service.runner import _EngineReady
from hal.netplay_service.runner import _InferenceProcessConfig
from hal.netplay_service.runner import _prepare_netplay_engine


@pytest.fixture(scope="module")
def prepared_hardware() -> tuple[InferenceEngine, _EngineReady, int]:
    if os.environ.get("HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 on the production GPU")
    value = os.environ.get("HAL_NETPLAY_POLICY")
    if not value or not Path(value).is_file():
        pytest.fail("HAL_NETPLAY_POLICY must name a qualified 059 capability-v2 bundle")
    capacity = int(os.environ.get("HAL_NETPLAY_QUALIFIED_CAPACITY", "1"))
    if capacity < 1:
        pytest.fail("HAL_NETPLAY_QUALIFIED_CAPACITY must be positive")
    hardware = torch.cuda.get_device_name()
    if "RTX 6000 Ada" in hardware and capacity < 2:
        pytest.fail("Ada netplay qualification requires at least two concurrent streams")
    pairs: list[tuple[Connection, Connection]] = [Pipe() for _ in range(capacity)]
    try:
        configure_inference_process()
        prepared = _prepare_netplay_engine(
            _InferenceProcessConfig(
                Path(value), "cuda", 0, True, capacity, 0.0005, "0" * 16, "0" * 40, "0" * 64, None
            ),
            {slot: parent for slot, (parent, _) in enumerate(pairs)},
        )
        yield prepared.engine, prepared.ready, capacity
    finally:
        for parent, child in pairs:
            parent.close()
            child.close()


def test_declared_profiles_meet_complete_path_budget(
    prepared_hardware: tuple[InferenceEngine, _EngineReady, int],
) -> None:
    _, ready, capacity = prepared_hardware
    assert len(ready.budgets) == 2
    for check in ready.budgets:
        assert len(check.measurements) == 1
        seconds = sorted(check.measurements[0].seconds)
        assert len(seconds) >= 200
        p95 = seconds[math.ceil(0.95 * len(seconds)) - 1]
        p99 = seconds[math.ceil(0.99 * len(seconds)) - 1]
        assert p99 < 1 / 60
        if "RTX 3060" in ready.hardware and capacity == 1 and check.timings[0].physical_delay_frames == 2:
            assert p95 <= 0.012


def test_prepared_profiles_do_not_compile_or_capture_during_play(
    prepared_hardware: tuple[InferenceEngine, _EngineReady, int],
) -> None:
    engine, ready, capacity = prepared_hardware
    for check in ready.budgets:
        timing = check.timings[0]
        profile = next(
            profile for profile in ready.profiles if profile.fixed_prefix_frames == timing.fixed_prefix_frames
        )
        policy = engine.profiles[profile]
        policy.reset_prediction()
        before = (set(policy._update_calls), set(policy._decoder_calls))
        with torch.compiler.set_stance("fail_on_recompile"):
            for sequence in range(30):
                source = ready.context_frames + sequence * timing.replan_interval_frames
                requests = tuple(
                    PredictionRequest(
                        100 + slot,
                        1,
                        sequence,
                        source,
                        make_warmup_observations(
                            policy.spec,
                            ready.context_frames if sequence == 0 else timing.replan_interval_frames,
                            100 + slot,
                            source,
                            timing.physical_delay_frames,
                            reset_first=sequence == 0,
                        ),
                        (NEUTRAL_CONTROLLER_ACTION,) * timing.fixed_prefix_frames,
                    )
                    for slot in range(capacity)
                )
                plans = policy.predict(requests)
                for request, plan in zip(requests, plans, strict=True):
                    validate_action_plan(request, plan, timing.prediction_horizon_frames)
        assert (set(policy._update_calls), set(policy._decoder_calls)) == before
