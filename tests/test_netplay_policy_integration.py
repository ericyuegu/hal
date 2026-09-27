"""Opt-in full-game qualification of the production netplay policy path."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing import Pipe
from pathlib import Path

import melee
import numpy as np
import pytest
import torch

from hal.eval.match_summary import summarize_trajectory
from hal.eval.netplay import run_netplay_match
from hal.eval.replays import require_completed_replay
from hal.eval.results import PlayResult
from hal.eval.scheduling import FrameTiming
from hal.inference.api import RuntimeConfig
from hal.inference.client import InferenceClient
from hal.inference.engine import InferenceEngine
from hal.inference.engine import configure_inference_process
from hal.netplay_service.replays import soak_replay_directory
from hal.netplay_service.runner import _InferenceProcessConfig
from hal.netplay_service.runner import _prepare_netplay_engine
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup


@dataclass(slots=True)
class _Harness:
    runtime: RuntimeConfig
    timings: tuple[FrameTiming, ...]
    clients: tuple[InferenceClient, InferenceClient]
    batcher: InferenceEngine
    stop: threading.Event
    engine: threading.Thread
    errors: list[BaseException]


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"set {variable} for the netplay policy integration test")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{variable} does not name a file")
    return path


def _required_code(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"set {name} for the netplay policy integration test")
    return value


@pytest.fixture(scope="module")
def policy_harness() -> _Harness:
    if os.environ.get("HAL_REQUIRE_NETPLAY_POLICY_INTEGRATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_POLICY_INTEGRATION=1 with two Slippi accounts")
    policy_path = _required_path("HAL_NETPLAY_POLICY")
    compiled = os.environ.get("HAL_NETPLAY_COMPILED", "0") == "1"
    parent_0, child_0 = Pipe()
    parent_1, child_1 = Pipe()
    try:
        configure_inference_process()
        batcher, ready = _prepare_netplay_engine(
            _InferenceProcessConfig(policy_path, "cuda", 0, compiled, 2, 0.0005),
            {0: parent_0, 1: parent_1},
        )
    except BaseException:
        for connection in (parent_0, child_0, parent_1, child_1):
            connection.close()
        raise
    runtime = RuntimeConfig(2, (2, 3), replan_interval_frames=4)
    stop = threading.Event()
    lost = threading.Event()
    prepared = {profile.fixed_prefix_frames: profile for profile in ready.profiles}
    clients = (
        InferenceClient(ready.spec, ready.context_frames, child_0, lost, prepared),
        InferenceClient(ready.spec, ready.context_frames, child_1, lost, prepared),
    )
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            with torch.compiler.set_stance("fail_on_recompile"):
                batcher.serve(stop)
        except BaseException as error:
            errors.append(error)
            lost.set()

    engine = threading.Thread(target=serve, daemon=True)
    engine.start()
    harness = _Harness(
        runtime, tuple(check.timings[0] for check in ready.budgets), clients, batcher, stop, engine, errors
    )
    try:
        yield harness
    finally:
        stop.set()
        for client in clients:
            client.close()
        engine.join(timeout=2)
        for connection in (parent_0, child_0, parent_1, child_1):
            connection.close()


@pytest.mark.integration
@pytest.mark.parametrize("delay", [2, 3])
def test_checkpoint_completes_batched_self_play(
    delay: int,
    policy_harness: _Harness,
) -> None:
    if delay not in policy_harness.runtime.transport_delays:
        pytest.skip("transport delay is not supported by this checkpoint")
    account_1 = _required_path("HAL_NETPLAY_USER_JSON_1")
    account_2 = _required_path("HAL_NETPLAY_USER_JSON_2")
    code_1 = _required_code("HAL_NETPLAY_CONNECT_CODE_1")
    code_2 = _required_code("HAL_NETPLAY_CONNECT_CODE_2")
    assert not account_1.samefile(account_2)
    assert code_1 != code_2
    before_calls = policy_harness.batcher.batch_calls

    with soak_replay_directory() as replay_root, ExitStack() as stack:
        replay_dirs = (replay_root / "account-1", replay_root / "account-2")
        for replay_dir in replay_dirs:
            replay_dir.mkdir()
        sessions = tuple(
            stack.enter_context(
                NetplaySession(
                    ISO_PATH,
                    dolphin_path=NETPLAY_EMULATOR_PATH,
                    user_json_path=account,
                    online_delay=delay,
                    replay_dir=replay_dir,
                    slippi_port=port,
                    connect_timeout_seconds=120,
                    realtime=True,
                )
            )
            for account, replay_dir, port in zip(
                (account_1, account_2),
                replay_dirs,
                (51441, 51442),
                strict=True,
            )
        )
        setups = (
            NetplaySetup(melee.Character.FOX, code_2, costume=0),
            NetplaySetup(melee.Character.FOX, code_1, costume=1),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    run_netplay_match,
                    session,
                    setup,
                    client,
                    policy_harness.runtime,
                    next(t for t in policy_harness.timings if t.input_delay_frames == delay),
                    player_identity="IBDW#0",
                    max_frames=54_000,
                    stream_id=slot,
                )
                for slot, (session, setup, client) in enumerate(
                    zip(sessions, setups, policy_harness.clients, strict=True)
                )
            ]
            results = tuple(future.result(timeout=900) for future in futures)

        replays = tuple(require_completed_replay(replay_dir, ()) for replay_dir in replay_dirs)
        assert all(replay.stat().st_size > 0 for replay in replays)
        timing = next(schedule for schedule in policy_harness.timings if schedule.input_delay_frames == delay)
        _assert_gameplay(results, timing.thinking_allowance_frames * 1000 / 60)

    assert not policy_harness.errors
    assert policy_harness.batcher.batch_calls > before_calls
    assert policy_harness.batcher.max_batch_items == 2


def _assert_gameplay(results: tuple[PlayResult, PlayResult], limit_ms: float) -> None:
    for result in results:
        summary = summarize_trajectory(result.trajectory)
        assert min(summary.p1_stocks_left, summary.p2_stocks_left) == 0
        assert result.inference_p95_ms < limit_ms
        assert result.game_fps >= 59.0
        assert result.frame_interval_p95_ms <= 20.0
        position = result.trajectory.post[result.ego_port]["position_x"]
        assert np.nanmax(position) - np.nanmin(position) > 5.0
