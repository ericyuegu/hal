"""Opt-in two-account calibration of Slippi's direct-connect transport delay."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import melee
import peppi_py
import pytest

from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.inputs import ControllerInputsValue
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.wire import peppi_port_to_libmelee

_NEUTRAL = ControllerInputsValue(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
_RIGHT = ControllerInputsValue(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def _account(name: str) -> tuple[Path, str]:
    user_json = os.environ.get(f"HAL_NETPLAY_USER_JSON_{name}")
    connect_code = os.environ.get(f"HAL_NETPLAY_CONNECT_CODE_{name}")
    if not user_json or not connect_code:
        pytest.fail(f"set HAL_NETPLAY_USER_JSON_{name} and HAL_NETPLAY_CONNECT_CODE_{name}")
    path = Path(user_json)
    if not path.is_file():
        pytest.fail(f"Slippi account file {name} does not exist")
    return path, connect_code


def _step_both(
    pool: ThreadPoolExecutor,
    sessions: tuple[NetplaySession, NetplaySession],
    inputs: tuple[ControllerInputsValue, ControllerInputsValue],
) -> tuple[dict, dict]:
    futures = [pool.submit(session.step, value) for session, value in zip(sessions, inputs, strict=True)]
    results = [future.result(timeout=30) for future in futures]
    assert all(in_game for _, in_game in results)
    return results[0][0], results[1][0]


def _stick_x(frame: dict, port: int) -> float:
    return float(frame["ports"][port]["leader"]["pre"]["joystick"]["x"])


def _replay_stick_x(replay: Path, port: int) -> dict[int, float]:
    game = peppi_py.read_slippi(str(replay), skip_frames=False)
    assert game.frames is not None
    indices = [index for index, player in enumerate(game.start.players) if peppi_port_to_libmelee(player.port) == port]
    assert len(indices) == 1
    frame_ids = game.frames.id.to_pylist()
    stick_x = game.frames.ports[indices[0]].leader.pre.joystick.x.to_pylist()
    # Later rows are the committed correction when rollback repeats a frame ID.
    return {int(frame_id): float(value) for frame_id, value in zip(frame_ids, stick_x, strict=True)}


@pytest.mark.integration
@pytest.mark.parametrize("delay", [2, 3])
def test_two_account_impulse_timing_replays_and_cleanup(
    tmp_path: Path,
    delay: int,
) -> None:
    if os.environ.get("HAL_REQUIRE_NETPLAY_INTEGRATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_INTEGRATION=1 with two Slippi accounts")
    account_1, code_1 = _account("1")
    account_2, code_2 = _account("2")
    if account_1.samefile(account_2) or code_1 == code_2:
        pytest.fail("the netplay integration test requires two distinct Slippi accounts")

    replay_dirs = (tmp_path / "account-1", tmp_path / "account-2")
    for replay_dir in replay_dirs:
        replay_dir.mkdir()
    sessions = (
        NetplaySession(
            ISO_PATH,
            dolphin_path=NETPLAY_EMULATOR_PATH,
            user_json_path=account_1,
            online_delay=delay,
            replay_dir=replay_dirs[0],
            slippi_port=51441,
        ),
        NetplaySession(
            ISO_PATH,
            dolphin_path=NETPLAY_EMULATOR_PATH,
            user_json_path=account_2,
            online_delay=delay,
            replay_dir=replay_dirs[1],
            slippi_port=51442,
        ),
    )
    setups = (
        NetplaySetup(melee.Character.FOX, code_2, costume=0),
        NetplaySetup(melee.Character.FOX, code_1, costume=1),
    )
    processes = []
    samples: list[list[tuple[int, float]]] = [[], []]
    source_frame_id = -1
    source_port = -1

    with sessions[0], sessions[1], ThreadPoolExecutor(max_workers=2) as pool:
        starts = [pool.submit(session.start_match, setup) for session, setup in zip(sessions, setups, strict=True)]
        for future in starts:
            future.result(timeout=600)

        assert sessions[0].ego_port == sessions[1].opponent_port
        assert sessions[1].ego_port == sessions[0].opponent_port
        assert sessions[0].ego_port is not None
        source_port = sessions[0].ego_port

        for session in sessions:
            if session._console is None or session._console._process is None:
                pytest.fail("Dolphin did not start")
            if session._console._process.poll() is not None:
                pytest.fail("Dolphin exited before the timing measurement")
            processes.append(session._console._process)

        for _ in range(240):
            frames = _step_both(pool, sessions, (_NEUTRAL, _NEUTRAL))
            frame_ids = tuple(int(frame["id"]) for frame in frames)
            values = tuple(_stick_x(frame, source_port) for frame in frames)
            if frame_ids[0] == frame_ids[1] and frame_ids[0] >= 0 and all(abs(value) < 0.01 for value in values):
                source_frame_id = frame_ids[0]
                for client, value in enumerate(values):
                    samples[client].append((source_frame_id, value))
                break
        else:
            pytest.fail("the two clients did not reach one synchronized neutral game frame")

        # This controller call is for the frame returned by the first step,
        # S_t+1. Slippi delay D must put it in S_t+1+D on both clients.
        landing_delta = delay + 1
        for returned_offset in range(1, landing_delta + 2):
            local_input = _RIGHT if returned_offset == 1 else _NEUTRAL
            frames = _step_both(pool, sessions, (local_input, _NEUTRAL))
            for client, frame in enumerate(frames):
                samples[client].append((int(frame["id"]), _stick_x(frame, source_port)))

        # Slippi writes replays on another thread. Keep the game alive until
        # the measured frames are durable before cleanup repairs the open file.
        for _ in range(120):
            _step_both(pool, sessions, (_NEUTRAL, _NEUTRAL))

    for session in sessions:
        assert session._console is None
        assert session._controller is None
        assert session._menu_helper is None
        assert session.ego_port is None
        assert session.opponent_port is None
    for process in processes:
        assert process.poll() is not None

    landing_frame_id = source_frame_id + landing_delta
    expected_frame_ids = list(range(source_frame_id, landing_frame_id + 2))
    for client_samples in samples:
        frame_ids = [frame_id for frame_id, _ in client_samples]
        values = [value for _, value in client_samples]
        assert frame_ids == expected_frame_ids
        assert [index for index, value in enumerate(values[1:]) if value > 0.9] == [delay]
        assert frame_ids[delay + 1] == landing_frame_id
        assert values[: delay + 1] == pytest.approx([0.0] * (delay + 1), abs=0.01)
        assert values[delay + 1] > 0.9
        assert values[delay + 2] == pytest.approx(0.0, abs=0.01)

    for client, replay_dir in enumerate(replay_dirs):
        replays = list(replay_dir.rglob("*.slp"))
        assert len(replays) == 1
        replay_values = _replay_stick_x(replays[0], source_port)
        for frame_id, live_value in samples[client]:
            assert replay_values[frame_id] == pytest.approx(live_value, abs=1e-6)
        replay_impulses = [frame_id for frame_id in expected_frame_ids if replay_values[frame_id] > 0.9]
        assert replay_impulses == [landing_frame_id]


@pytest.mark.integration
@pytest.mark.parametrize("delay", [2, 3])
def test_realtime_dolphin_advances_during_inference_and_delivery(tmp_path: Path, delay: int) -> None:
    """Qualify frame placement with inference and response-delivery overruns."""
    import json
    import threading
    import time
    from multiprocessing import Pipe

    from hal.controller import NEUTRAL_CONTROLLER_ACTION
    from hal.controller import ControllerAction
    from hal.inference.api import PolicyInput
    from hal.inference.api import PolicySpec
    from hal.inference.chunks import TimingSchedule
    from hal.inference.chunks import chunk_response
    from hal.netplay_service.chunks import ChunkBatcher
    from hal.netplay_service.chunks import RemoteChunkPolicy
    from hal.netplay_service.schedule import FrameSchedule
    from hal.sim.inputs import canonical_pre_to_action
    from hal.sim.inputs import controller_actions_match

    if os.environ.get("HAL_REQUIRE_NETPLAY_INTEGRATION") != "1":
        pytest.skip("set HAL_REQUIRE_NETPLAY_INTEGRATION=1 with two Slippi accounts")
    account_1, code_1 = _account("1")
    account_2, code_2 = _account("2")
    assert not account_1.samefile(account_2) and code_1 != code_2
    timing = TimingSchedule(3, delay, 12)
    stop = threading.Event()
    lost = threading.Event()
    errors = []
    pairs = [Pipe(), Pipe()]

    class DelayedPolicy:
        spec = PolicySpec("timing qualification", "test", (), (2, 3))
        context_frames = 16
        supported_horizons = (12,)
        sampling_seed = 0

        def plan_chunks(self, requests):
            # Periodic overruns exceed B=4; ordinary calls include 1.2 frames
            # of computation and .6 frames of delivery delay below.
            time.sleep(0.100 if requests[0].sequence % 7 == 3 else 0.020)
            return tuple(
                chunk_response(
                    request,
                    tuple(
                        ControllerAction(0.8 if (request.source_frame + offset) % 24 < 12 else -0.8, 0, 0, 0, 0, 0, 0)
                        for offset in range(len(request.forced_prefix) + 1, 13)
                    ),
                )
                for request in requests
            )

    class DeliveryConnection:
        def __init__(self, connection):
            self.connection = connection

        def recv(self):
            return self.connection.recv()

        def send(self, response):
            time.sleep(0.010)
            self.connection.send(response)

        def fileno(self):
            return self.connection.fileno()

    policy = DelayedPolicy()
    batcher = ChunkBatcher(
        policy, 12, {slot: DeliveryConnection(pair[0]) for slot, pair in enumerate(pairs)}, batch_wait_seconds=0.0005
    )
    clients = [RemoteChunkPolicy(policy.spec, 16, (timing,), pair[1], lost) for pair in pairs]

    def serve():
        try:
            batcher.serve(stop)
        except BaseException as error:
            errors.append(error)
            lost.set()

    engine = threading.Thread(target=serve, daemon=True)
    engine.start()
    replay_dirs = (tmp_path / "realtime-1", tmp_path / "realtime-2")
    sessions = tuple(
        NetplaySession(
            ISO_PATH,
            dolphin_path=NETPLAY_EMULATOR_PATH,
            user_json_path=account,
            online_delay=delay,
            replay_dir=replay_dir,
            slippi_port=51441 + slot,
            realtime=True,
        )
        for slot, (account, replay_dir) in enumerate(zip((account_1, account_2), replay_dirs, strict=True))
    )
    setups = (
        NetplaySetup(melee.Character.FOX, code_2, costume=0),
        NetplaySetup(melee.Character.FOX, code_1, costume=1),
    )

    def play(slot, first):
        session, client = sessions[slot], clients[slot]
        assert session.ego_port is not None
        schedule = FrameSchedule(timing, 16, client.start_match())
        frames = [first]
        observed = {}
        submitted = {}
        advanced_while_waiting = 0
        latencies = []
        timestamps = []
        while frames[-1]["id"] < 360:
            for frame in frames:
                applied = canonical_pre_to_action(frame["ports"][session.ego_port]["leader"]["pre"])
                observed[frame["id"]] = applied.main_x
                schedule.observe(
                    PolicyInput(slot, frame["id"], session.ego_port, {}, applied, (NEUTRAL_CONTROLLER_ACTION,) * delay)
                )
                advanced_while_waiting += client.busy
            frame_id = frames[-1]["id"]
            response = client.poll()
            if response is not None:
                schedule.receive(response)
                latencies.append(client.last_latency)
            schedule.handoff(frame_id)
            if not client.busy:
                request = schedule.begin_request()
                if request is not None:
                    client.submit(request)
            selected = schedule.submit(frame_id)
            submitted[frame_id + delay + 1] = selected
            session.submit(selected)
            frames, live = session.read_frames()
            timestamps.extend(session.frame_times)
            assert live
        matches = sum(
            controller_actions_match(expected, ControllerAction(observed[frame], 0, 0, 0, 0, 0, 0))
            for frame, expected in submitted.items()
            if frame in observed
        )
        targets = sum(frame in observed for frame in submitted)
        elapsed = timestamps[-1] - timestamps[0]
        metrics = {
            "fps": (len(timestamps) - 1) / elapsed,
            "frame_intervals_seconds": [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)],
            "request_seconds": latencies,
            "deadline_misses": schedule.deadline_misses,
            "transport_corrections": schedule.transport_corrections,
            "advanced_while_waiting": advanced_while_waiting,
            "matched_targets": matches,
            "checked_targets": targets,
        }
        (tmp_path / f"realtime-{slot}-metrics.json").write_text(json.dumps(metrics))
        assert advanced_while_waiting > 60
        assert schedule.deadline_misses > 0
        assert metrics["fps"] >= 59
        assert any(abs(value) > 0.5 for value in observed.values())
        assert matches / targets > 0.9
        return session.ego_port, observed

    try:
        with sessions[0], sessions[1], ThreadPoolExecutor(max_workers=2) as pool:
            starts = [pool.submit(session.start_match, setup) for session, setup in zip(sessions, setups, strict=True)]
            first = [future.result(timeout=600) for future in starts]
            futures = [pool.submit(play, slot, frame) for slot, frame in enumerate(first)]
            results = [future.result(timeout=60) for future in futures]
        for replay_dir, (port, observations) in zip(replay_dirs, results, strict=True):
            replays = list(replay_dir.rglob("*.slp"))
            assert len(replays) == 1
            replay_values = _replay_stick_x(replays[0], port)
            for frame, value in observations.items():
                assert replay_values[frame] == pytest.approx(value, abs=1e-6)
        assert not errors
    finally:
        stop.set()
        engine.join(2)
        for pair in pairs:
            for connection in pair:
                connection.close()
