import hashlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import melee
import melee.console
import pytest

import hal.sim.netplay as netplay
from hal.sim.inputs import ControllerInputsValue
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.session import fix_dolphin_ini_case


def _session(tmp_path: Path, **kwargs) -> NetplaySession:
    return NetplaySession(
        "/fake/game.ciso",
        dolphin_path="/fake/dolphin",
        user_json_path="/fake/user.json",
        online_delay=2,
        replay_dir=tmp_path,
        **kwargs,
    )


def _live(*, ego_port: int = 1, character: melee.Character = melee.Character.FOX):
    opponent_port = 3 - ego_port
    players = {
        ego_port: SimpleNamespace(character=character, costume=0, connectCode="BOT#0"),
        opponent_port: SimpleNamespace(character=character, costume=0, connectCode="HUMAN#1"),
    }
    return SimpleNamespace(menu_state=melee.Menu.IN_GAME, players=players, stage=melee.Stage.BATTLEFIELD)


def _dolphin_version() -> melee.console.DolphinVersion:
    return melee.console.DolphinVersion(False, "3.6.4", melee.console.DolphinBuild.NETPLAY)


def test_console_uses_blocking_uncapped_non_exi_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    console_type = Mock()
    console_type.return_value._get_dolphin_config_path.return_value = str(tmp_path / "missing")
    version = _dolphin_version()
    original_probe = melee.console.get_dolphin_version
    console_type.side_effect = lambda **_kwargs: (
        console_type.return_value
        if melee.console.get_dolphin_version("ignored") is version
        else pytest.fail("Console construction did not use the validated Dolphin version")
    )
    monkeypatch.setattr("hal.sim.netplay._tested_dolphin_version", lambda _path: version)
    monkeypatch.setattr("hal.sim.netplay.melee.Console", console_type)
    monkeypatch.setattr("hal.sim.netplay.teardown_console", Mock())
    with _session(tmp_path):
        pass
    assert melee.console.get_dolphin_version is original_probe
    kwargs = console_type.call_args.kwargs
    assert kwargs["blocking_input"] is True
    assert kwargs["polling_mode"] is True
    assert kwargs["polling_timeout"] == 30.0
    assert kwargs["skip_rollback_frames"] is True
    assert kwargs["rollback_resolution"] == "first"
    assert kwargs["gfx_backend"] == "Vulkan"
    assert kwargs["emulation_speed"] == 0
    assert kwargs["use_exi_inputs"] is False
    assert kwargs["enable_ffw"] is False
    assert kwargs["online_delay"] == 2
    assert kwargs["replay_monthly_folders"] is False


def test_controller_is_created_before_dolphin_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    console = Mock()
    console.connect.side_effect = lambda: events.append("console-connect") or True

    class Controller:
        def __init__(self, **_kwargs) -> None:
            events.append("controller")

        def connect(self) -> bool:
            events.append("controller-connect")
            return True

        def release_all(self) -> None:
            events.append("neutral")

        def flush(self) -> None:
            events.append("flush")

    console.run.side_effect = lambda **_kwargs: events.append("launch")
    monkeypatch.setattr("hal.sim.netplay.melee.Controller", Controller)
    monkeypatch.setattr("hal.sim.netplay.fix_dolphin_ini_case", lambda _console: events.append("fix"))
    monkeypatch.setattr("hal.sim.netplay.popen_with_pdeathsig", nullcontext)
    monkeypatch.setattr("hal.sim.netplay.step_blocking", lambda *_args: _live())
    monkeypatch.setattr("hal.sim.netplay.canonical_frame", lambda _state: {"id": 0})
    session = _session(tmp_path)
    session._console = console
    session.start_match(NetplaySetup(melee.Character.FOX, "HUMAN#1"))
    assert events == ["controller", "fix", "launch", "controller-connect", "neutral", "flush", "console-connect"]


def test_character_selection_is_passed_to_menu_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection = []
    helper = Mock()
    helper.menu_helper_simple.side_effect = lambda **kwargs: selection.append(kwargs["character_selected"])
    session = _session(tmp_path)
    session._console = Mock()
    session._controller = Mock()
    session._menu_helper = helper
    states = iter(
        [
            SimpleNamespace(menu_state=melee.Menu.CHARACTER_SELECT),
            _live(character=melee.Character.FALCO),
        ]
    )
    monkeypatch.setattr("hal.sim.netplay.step_blocking", lambda *_args: next(states))
    monkeypatch.setattr("hal.sim.netplay.canonical_frame", lambda _state: {"id": 0})
    session._navigate_to_live(NetplaySetup(melee.Character.FALCO, "HUMAN#1"))
    assert selection == [melee.Character.FALCO]


def test_main_menu_uses_libmelee_direct_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path)
    session._console = Mock()
    session._controller = controller = Mock()
    session._menu_helper = helper = Mock()
    states = iter(
        [
            SimpleNamespace(
                menu_state=melee.Menu.MAIN_MENU,
                submenu=melee.SubMenu.ONLINE_PLAY_SUBMENU,
                menu_selection=1,
                frame=1,
            ),
            _live(),
        ]
    )
    monkeypatch.setattr("hal.sim.netplay.step_blocking", lambda *_args: next(states))
    monkeypatch.setattr("hal.sim.netplay.canonical_frame", lambda _state: {"id": 0})
    session._navigate_to_live(NetplaySetup(melee.Character.FOX, "HUMAN#1"))
    helper.choose_direct_online.assert_called_once()
    assert helper.choose_direct_online.call_args.args[1] is controller


def test_unknown_dolphin_build_is_rejected(tmp_path: Path) -> None:
    executable = tmp_path / "Slippi.AppImage"
    executable.write_bytes(b"not the tested build")
    with pytest.raises(RuntimeError, match="requires the tested Slippi 3.6.4 Linux AppImage"):
        netplay._tested_dolphin_version(str(executable))


def test_tested_dolphin_fingerprint_returns_pinned_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "Slippi.AppImage"
    executable.write_bytes(b"tested build")
    monkeypatch.setattr(netplay, "_SLIPPI_3_6_4_LINUX_SHA256", hashlib.sha256(b"tested build").hexdigest())
    version = netplay._tested_dolphin_version(str(executable))
    assert version == _dolphin_version()


def test_opponent_code_disambiguates_same_character_ports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("hal.sim.netplay.melee.gamestate.port_detector", lambda *_args: 1)
    session = _session(tmp_path)
    session._discover_ports(_live(ego_port=2), NetplaySetup(melee.Character.FOX, "HUMAN#1"))
    assert (session.ego_port, session.opponent_port) == (2, 1)


def test_same_character_ports_without_identity_are_rejected(tmp_path: Path) -> None:
    gamestate = _live()
    for player in gamestate.players.values():
        player.connectCode = ""
    with pytest.raises(RuntimeError, match="both players use FOX"):
        _session(tmp_path)._discover_ports(gamestate, NetplaySetup(melee.Character.FOX, "HUMAN#1"))


def test_character_disambiguates_ports_with_the_same_connect_code(tmp_path: Path) -> None:
    gamestate = _live(ego_port=2)
    gamestate.players[1].connectCode = "SAME#1"
    gamestate.players[2].connectCode = "SAME#1"
    gamestate.players[1].character = melee.Character.FALCO
    session = _session(tmp_path)
    session._discover_ports(gamestate, NetplaySetup(melee.Character.FOX, "SAME#1"))
    assert (session.ego_port, session.opponent_port) == (2, 1)


def test_step_requires_an_explicit_input_even_when_neutral(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session._console = Mock()
    session._controller = Mock()
    with pytest.raises(TypeError, match="requires controller inputs"):
        session.step(None)

    neutral = ControllerInputsValue(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    assert neutral.main_x == 0.0


@pytest.mark.parametrize("frame_id", [9, 10, 12])
def test_step_rejects_a_nonconsecutive_live_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frame_id: int,
) -> None:
    session = _session(tmp_path)
    session._console = Mock()
    session._controller = Mock()
    session._last_frame_id = 10
    state = SimpleNamespace(menu_state=melee.Menu.IN_GAME, frame=frame_id)
    monkeypatch.setattr("hal.sim.netplay.apply_inputs", Mock())
    monkeypatch.setattr("hal.sim.netplay.step_blocking", lambda *_args: state)
    monkeypatch.setattr("hal.sim.netplay.canonical_frame", lambda value: {"id": value.frame})

    with pytest.raises(RuntimeError, match="rollback filtering did not preserve call alignment"):
        session.step(ControllerInputsValue(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0))


def test_teardown_uses_shared_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup = Mock()
    monkeypatch.setattr("hal.sim.netplay.teardown_console", cleanup)
    session = _session(tmp_path)
    session._console = console = Mock()
    session._teardown()
    cleanup.assert_called_once_with(console, str(tmp_path))
    assert session._console is None


def test_dolphin_ini_keys_are_restored_to_camel_case(tmp_path: Path) -> None:
    config = tmp_path / "Config"
    config.mkdir()
    ini = config / "Dolphin.ini"
    ini.write_text(
        "[Core]\nslippionlinedelay = 2\nblockingpipes = True\nslippireplaydir = /tmp/replays\nsidevice0 = 6\n"
    )
    console = SimpleNamespace(_get_dolphin_config_path=lambda: str(config))
    fix_dolphin_ini_case(console)
    assert ini.read_text() == (
        "[Core]\nSlippiOnlineDelay = 2\nBlockingPipes = True\nSlippiReplayDir = /tmp/replays\nSIDevice0 = 6\n"
    )
