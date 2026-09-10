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
