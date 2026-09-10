"""Opt-in latency qualification for the production compiled policy shape."""

import math
import os
import time
from pathlib import Path

import pytest
import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import Policy
from hal.inference.api import PolicyInput
from hal.inference.api import RuntimeConfig
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


def _measure(policy: Policy, rows: int, delay: int, frames: int = 300) -> float:
    seconds = []
    fields = policy.spec.required_observation_fields
    for frame_id in range(frames):
        inputs = tuple(_input(fields, stream_id, frame_id, delay) for stream_id in range(rows))
        started = time.perf_counter()
        policy.step(inputs)
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
    policy = load_policy(value, device="cuda", seed=0, compiled=True)
    policy.prepare(RuntimeConfig(max_batch_size=2, transport_delays=(2, 3)))

    with torch.compiler.set_stance("fail_on_recompile"):
        p95_ms = _measure(policy, rows, delay)

    assert p95_ms < limit_ms
