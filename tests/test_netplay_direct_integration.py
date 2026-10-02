"""Slippi 3.6.4 direct mode across consecutive games, with two local Dolphins."""

import os
import threading
from pathlib import Path

import melee
import peppi_py
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.netplay_service.domain import account_connect_code
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import DirectMenuDriver
from hal.sim.netplay import DirectSelection
from hal.sim.netplay import MenuDriver
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.netplay import PlayerDisconnected

pytestmark = pytest.mark.integration
# Replays use Melee's external stage IDs, not libmelee's internal ones.
_STADIUM_REPLAY_ID = 3

_WALK_LEFT = ControllerAction(
    main_x=-1.0,
    main_y=0.0,
    c_x=0.0,
    c_y=0.0,
    trigger_l=0.0,
    trigger_r=0.0,
    buttons=0,
)


def _accounts() -> tuple[Path, Path]:
    bot = os.environ.get("HAL_NETPLAY_BOT_ACCOUNT")
    peer = os.environ.get("HAL_NETPLAY_PEER_ACCOUNT")
    if not bot or not peer:
        if os.environ.get("HAL_REQUIRE_INTEGRATION") == "1":
            pytest.fail("HAL_NETPLAY_BOT_ACCOUNT and HAL_NETPLAY_PEER_ACCOUNT are required")
        pytest.skip("two Slippi accounts are required")
    return Path(bot), Path(peer)


def _session(
    account: Path,
    port: int,
    replay_dir: Path,
    driver: MenuDriver | None = None,
) -> NetplaySession:
    return NetplaySession(
        ISO_PATH,
        dolphin_path=NETPLAY_EMULATOR_PATH,
        user_json_path=account,
        online_delay=2,
        replay_dir=replay_dir,
        slippi_port=port,
        realtime=True,
        graphics_backend="OGL",
        connect_timeout_seconds=600,
        step_timeout_seconds=60,
        menu_driver=driver,
    )


def _play_out(session: NetplaySession, action: ControllerAction) -> None:
    while True:
        session.submit(action)
        _frames, in_game = session.read_frames()
        if not in_game:
            return


def test_direct_mode_session(tmp_path: Path) -> None:
    bot_account, peer_account = _accounts()
    bot_code = account_connect_code(bot_account)
    peer_code = account_connect_code(peer_account)
    selection = [DirectSelection(1, melee.Character.FOX, 0, melee.Stage.POKEMON_STADIUM, "IBDW#0")]
    locks = [0]
    driver = DirectMenuDriver(
        opponent_code=peer_code,
        selection=lambda: selection[-1],
        lock_requests=lambda: locks[-1],
        connect_timeout_seconds=120,
        idle_timeout_seconds=300,
        hold_seconds=5,
        hold_cap_seconds=30,
        probe_interval_seconds=2,
    )
    errors: list[BaseException] = []
    peer_done = threading.Event()

    def peer() -> None:
        try:
            with _session(peer_account, 51472, tmp_path / "peer") as session:
                setup = NetplaySetup(melee.Character.MARTH, bot_code, stage=melee.Stage.BATTLEFIELD)
                session.start_match(setup)
                _play_out(session, NEUTRAL_CONTROLLER_ACTION)
                session.start_rematch(setup)
                _play_out(session, NEUTRAL_CONTROLLER_ACTION)
                session.start_rematch(setup)
                _play_out(session, NEUTRAL_CONTROLLER_ACTION)
                peer_done.wait(120)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    with _session(bot_account, 51471, tmp_path / "bot", driver) as bot:
        for game in range(3):
            driver.begin_game()
            if game == 1:
                selection.append(
                    DirectSelection(
                        2,
                        melee.Character.FALCO,
                        0,
                        melee.Stage.POKEMON_STADIUM,
                        "IBDW#0",
                    )
                )
            start = bot.start_match if game == 0 else bot.start_rematch
            start(NetplaySetup(melee.Character.FOX, peer_code, local_code=bot_code))
            driver.connected = True
            assert driver.locked is not None
            assert driver.locked.revision == (1 if game == 0 else 2)
            _play_out(bot, _WALK_LEFT)
        driver.begin_game()
        peer_done.set()
        with pytest.raises(PlayerDisconnected):
            bot.start_rematch(NetplaySetup(melee.Character.FALCO, peer_code, local_code=bot_code))
    thread.join(60)
    assert not errors, errors
    replays = sorted((tmp_path / "bot").rglob("*.slp"))
    starts = [peppi_py.read_slippi(str(path), skip_frames=True).start for path in replays]
    stadium = [start for start in starts if start.stage == _STADIUM_REPLAY_ID]
    assert len(stadium) >= 2 and all(start.is_frozen_ps for start in stadium)
