"""Opt-in latency qualification for the production compiled policy shape."""

import math
import os
import time
from pathlib import Path

import pytest
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import PolicyInput
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.bundle import read_policy_manifest
from hal.inference.loader import load_policy


def _input(spec_fields: tuple[str, ...], stream_id: int, frame_id: int, delay: int) -> PolicyInput:
    return PolicyInput(
        stream_id=stream_id,
        frame_id=frame_id,
        controlled_port=1,
        observation={name: 0.0 for name in spec_fields},
        applied_action=NEUTRAL_CONTROLLER_ACTION,
        pending_actions=(NEUTRAL_CONTROLLER_ACTION,) * delay,
        player_identity="PLATINUM",
        reset=frame_id == 0,
    )


def _measure(policy: PredictionPolicy, rows: int, delay: int, frames: int = 300) -> float:
    seconds = []
    fields = policy.spec.required_observation_fields
    for frame_id in range(frames):
        inputs = tuple(_input(fields, stream_id, frame_id, delay) for stream_id in range(rows))
        started = time.perf_counter()
        requests = tuple(
            PredictionRequest(item.stream_id, 1, frame_id, frame_id, (item,), item.pending_actions) for item in inputs
        )
        policy.predict(requests)
        seconds.append(time.perf_counter() - started)
    seconds.sort()
    return 1_000.0 * seconds[math.ceil(0.95 * len(seconds)) - 1]


@pytest.mark.integration
@pytest.mark.parametrize(("delay", "limit_ms"), [(2, 33.3), (3, 16.7)])
@pytest.mark.parametrize("rows", [1, 2])
def test_compiled_policy_latency_has_no_post_prepare_recompile(delay: int, limit_ms: float, rows: int) -> None:
    if os.environ.get("HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 on the production GPU")
    value = os.environ.get("HAL_NETPLAY_POLICY")
    if not value or not Path(value).is_file():
        pytest.fail("HAL_NETPLAY_POLICY must name the production policy bundle")
    manifest = read_policy_manifest(value)
    if delay not in manifest.supported_transport_delays or (manifest.backend == "o59-history-decoder" and rows != 1):
        pytest.skip("this policy does not serve this delay or batch size")
    policy = load_policy(value, device="cuda", seed=0, compiled=True)
    assert isinstance(policy, PredictionPolicy)
    policy.prepare_prediction(RuntimeConfig(rows, (delay,)), policy.prediction_horizon, delay)

    with torch.compiler.set_stance("fail_on_recompile"):
        p95_ms = _measure(policy, rows, delay)

    assert p95_ms < limit_ms


@pytest.mark.integration
@pytest.mark.parametrize("rows", [1, 2])
def test_qualified_predictions_do_not_compile_during_play(rows: int) -> None:
    from hal.eval.qualification import check_realtime_budget
    from hal.inference.warmup import make_warmup_observations

    if os.environ.get("HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 on the production GPU")
    value = os.environ.get("HAL_NETPLAY_POLICY")
    if not value or not Path(value).is_file():
        pytest.fail("HAL_NETPLAY_POLICY must name the production policy bundle")
    manifest = read_policy_manifest(value)
    runtime = RuntimeConfig(rows, tuple(delay for delay in manifest.supported_transport_delays if delay in (2, 3)))
    policy = load_policy(value, device="cuda", seed=0, compiled=True)
    assert isinstance(policy, PredictionPolicy)
    result = check_realtime_budget(policy, runtime, 0.0005)
    with torch.compiler.set_stance("fail_on_recompile"):
        for timing in result.timings:
            policy.reset_prediction()
            for sequence in range(30):
                source = policy.context_frames + sequence * timing.replan_interval_frames
                requests = tuple(
                    PredictionRequest(
                        slot,
                        0,
                        sequence,
                        source,
                        make_warmup_observations(
                            policy.spec,
                            policy.context_frames if sequence == 0 else timing.replan_interval_frames,
                            slot,
                            source,
                            timing.input_delay_frames,
                            reset_first=sequence == 0,
                        ),
                        (NEUTRAL_CONTROLLER_ACTION,) * timing.fixed_prefix_frames,
                    )
                    for slot in range(rows)
                )
                responses = policy.predict(requests)
                assert all(
                    len(response.actions) == timing.prediction_horizon_frames - timing.fixed_prefix_frames
                    for response in responses
                )
