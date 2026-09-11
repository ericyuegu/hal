"""One local Dolphin session for Slippi direct-connect netplay."""

import atexit
import configparser
import hashlib
import threading
import time
from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import melee
import melee.console
from loguru import logger

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import apply_inputs
from hal.sim.inputs import canonical_pre_to_action
from hal.sim.inputs import controller_actions_match
from hal.sim.session import LIVE_MENU_STATES
from hal.sim.session import canonical_frame
from hal.sim.session import fix_dolphin_ini_case
from hal.sim.session import kill_dolphin
from hal.sim.session import popen_with_pdeathsig
from hal.sim.session import step_blocking
from hal.sim.session import teardown_console

_SLIPPI_3_6_4_LINUX_SHA256 = "e0f984e5bbecb98e3a746da1f173a475b06c3a1ba6b73e2e31bbe85a5f5a5e8a"
_DOLPHIN_VERSION_PATCH_LOCK = threading.RLock()
_NATIVE_EFB_SCALE = "2"


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _tested_dolphin_version(dolphin_path: str) -> melee.console.DolphinVersion:
    """Validate the exact GUI Slippi build tested for blocking netplay."""
    executable = Path(melee.console.get_exe_path(dolphin_path))
    try:
        with executable.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as error:
        raise RuntimeError(f"cannot read netplay Dolphin executable {executable}: {error}") from error
    if digest != _SLIPPI_3_6_4_LINUX_SHA256:
        raise RuntimeError(
            f"unsupported netplay Dolphin {executable} (sha256 {digest}); "
            "HAL netplay requires the tested Slippi 3.6.4 Linux AppImage"
        )
    return melee.console.DolphinVersion(
        mainline=False,
        version="3.6.4",
        build=melee.console.DolphinBuild.NETPLAY,
    )


def _set_native_internal_resolution(console: melee.Console) -> None:
    """Set native rendering in the isolated Slippi 3.6.4 profile."""
    config_path = Path(console._get_dolphin_config_path())
    config_path.mkdir(parents=True, exist_ok=True)
    ini_path = config_path / "GFX.ini"
    config = _CaseSensitiveConfigParser(interpolation=None)
    config.read(ini_path)
    if not config.has_section("Settings"):
        config.add_section("Settings")
    # Pinned libmelee 0.47.0 does not expose EFBScale. Slippi uses 2 for
    # native resolution and defaults to 4 (2x). Remove this when libmelee can
    # configure the internal resolution directly.
    config.set("Settings", "EFBScale", _NATIVE_EFB_SCALE)
    with ini_path.open("w") as output:
        config.write(output)


@contextmanager
def _known_dolphin_version(version: melee.console.DolphinVersion) -> Iterator[None]:
    """Bypass libmelee 0.47.0's hanging GUI ``--version`` probe.

    Remove this boundary workaround when the pinned libmelee either bounds its
    GUI version probe or accepts an already validated version.
    """
    with _DOLPHIN_VERSION_PATCH_LOCK:
        original = melee.console.get_dolphin_version

        def known_version(_path: str) -> melee.console.DolphinVersion:
            return version

        melee.console.__dict__["get_dolphin_version"] = known_version
        try:
            yield
        finally:
            melee.console.__dict__["get_dolphin_version"] = original


@dataclass(frozen=True, slots=True)
class NetplaySetup:
    """The local character, remote Slippi code, and selected stage."""

    character: melee.Character
    opponent_code: str
    costume: int = 0
    stage: melee.Stage = melee.Stage.FINAL_DESTINATION


class NetplaySession:
    """Drive one blocking, real-time Dolphin direct-connect session."""

    def __init__(
        self,
        iso_path: str | Path,
        *,
        dolphin_path: str | Path,
        user_json_path: str | Path,
        online_delay: int,
        replay_dir: str | Path,
        slippi_port: int = 51441,
        step_timeout_seconds: float = 30.0,
        connect_timeout_seconds: float = 600.0,
    ) -> None:
        if online_delay not in (2, 3):
            raise ValueError(f"online_delay must be 2 or 3, got {online_delay}")
        self.iso_path = str(iso_path)
        self.dolphin_path = str(dolphin_path)
        self.user_json_path = str(user_json_path)
        self.online_delay = online_delay
        self.replay_dir = str(Path(replay_dir).resolve())
        self.slippi_port = slippi_port
        self.step_timeout_seconds = step_timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.ego_port: int | None = None
        self.opponent_port: int | None = None
        self._console: melee.Console | None = None
        self._controller: melee.Controller | None = None
        self._menu_helper: melee.MenuHelper | None = None
        self._last_frame_id: int | None = None
        self._atexit_kill = self._kill_dolphin_only

    def __enter__(self) -> Self:
        version = _tested_dolphin_version(self.dolphin_path)
        with _known_dolphin_version(version):
            self._console = melee.Console(
                path=self.dolphin_path,
                slippi_port=self.slippi_port,
                online_delay=self.online_delay,
                user_json_path=self.user_json_path,
                blocking_input=True,
                polling_mode=True,
                # Console.step() flushes every controller before it polls. A
                # zero-timeout polling loop can therefore send several FLUSH
                # commands for one returned frame and destroy call alignment.
                polling_timeout=self.step_timeout_seconds,
                skip_rollback_frames=True,
                rollback_resolution="first",
                # Slippi 3.6.4 defaults to OpenGL, which stalls CUDA policy
                # work on NVIDIA. Vulkan keeps the tested D3 path below one
                # frame; revisit this with the executable fingerprint above.
                gfx_backend="Vulkan",
                disable_audio=True,
                tmp_home_directory=True,
                save_replays=True,
                replay_dir=self.replay_dir,
                replay_monthly_folders=False,
                emulation_speed=0,
                use_exi_inputs=False,
                enable_ffw=False,
            )
        fix_dolphin_ini_case(self._console)
        _set_native_internal_resolution(self._console)
        atexit.register(self._atexit_kill)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._teardown()

    def start_match(
        self,
        setup: NetplaySetup,
        *,
        on_countdown_frame: Callable[[dict], None] | None = None,
    ) -> dict:
        """Connect and emit countdown states before returning frame zero."""
        if self._console is None:
            raise RuntimeError("NetplaySession must be used as a context manager")
        if self._controller is not None:
            raise RuntimeError("NetplaySession has already launched Dolphin")
        if not setup.opponent_code:
            raise ValueError("opponent_code must be non-empty")
        logger.info(
            "starting Dolphin slippi_port={} delay={} character={} stage={} replay_dir={}",
            self.slippi_port,
            self.online_delay,
            setup.character.name,
            setup.stage.name,
            self.replay_dir,
        )
        # Controller construction creates configuration and the named pipe that
        # Dolphin reads during launch.
        self._controller = melee.Controller(
            console=self._console,
            port=1,
            type=melee.ControllerType.STANDARD,
            fix_analog_inputs=False,
        )
        fix_dolphin_ini_case(self._console)
        with popen_with_pdeathsig():
            self._console.run(iso_path=self.iso_path)
        if not self._controller.connect():
            raise RuntimeError("failed to connect the local controller")
        self._controller.release_all()
        self._controller.flush()
        if not self._console.connect():
            raise RuntimeError("failed to connect to Dolphin Slippi server")
        self._menu_helper = melee.MenuHelper()
        logger.info("waiting for direct-connect opponent {}", setup.opponent_code)
        first_frame = self._navigate_to_live(setup, on_countdown_frame=on_countdown_frame)
        frame_id = first_frame.get("id")
        if not isinstance(frame_id, int):
            raise RuntimeError(f"first netplay frame has invalid id {frame_id!r}")
        self._last_frame_id = frame_id
        return first_frame

    def start_rematch(
        self,
        setup: NetplaySetup,
        *,
        on_countdown_frame: Callable[[dict], None] | None = None,
    ) -> dict:
        """Navigate to a rematch and emit countdown states before frame zero."""
        if self._console is None or self._controller is None or self._menu_helper is None:
            raise RuntimeError("start_match must complete before start_rematch")
        if self._last_frame_id is not None:
            raise RuntimeError("the current netplay match has not ended")
        # MenuHelper latches its character and stage choices. A rematch can
        # change either one, so it needs a fresh navigation state machine.
        self._menu_helper = melee.MenuHelper()
        self._controller.release_all()
        self._controller.flush()
        first_frame = self._navigate_to_live(setup, on_countdown_frame=on_countdown_frame)
        frame_id = first_frame.get("id")
        if not isinstance(frame_id, int):
            raise RuntimeError(f"first netplay frame has invalid id {frame_id!r}")
        self._last_frame_id = frame_id
        return first_frame

    def park_menu(self) -> melee.Menu:
        """Send neutral input while a connected player decides on a rematch."""
        if self._console is None or self._controller is None:
            raise RuntimeError("start_match must complete before park_menu")
        if self._last_frame_id is not None:
            raise RuntimeError("cannot park the menu while a match is live")
        self._controller.release_all()
        self._controller.flush()
        gamestate = step_blocking(self._console, self.step_timeout_seconds)
        if gamestate.menu_state in LIVE_MENU_STATES:
            raise RuntimeError("netplay entered a game while waiting for a rematch")
        return gamestate.menu_state

    def _navigate_to_live(
        self,
        setup: NetplaySetup,
        *,
        on_countdown_frame: Callable[[dict], None] | None = None,
    ) -> dict:
        assert self._console is not None
        assert self._controller is not None
        assert self._menu_helper is not None
        deadline = time.monotonic() + self.connect_timeout_seconds
        last_status: tuple[melee.Menu, object] | None = None
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"did not reach IN_GAME within {self.connect_timeout_seconds:.0f}s "
                    "while waiting for the remote player"
                )
            gamestate = step_blocking(self._console, self.step_timeout_seconds)
            status = (
                gamestate.menu_state,
                getattr(gamestate, "submenu", None),
            )
            if status != last_status:
                logger.info(
                    f"netplay menu: {status[0].name}; submenu={getattr(status[1], 'name', status[1])}; "
                    f"selection={getattr(gamestate, 'menu_selection', None)}"
                )
                last_status = status
            if gamestate.menu_state in LIVE_MENU_STATES:
                self._discover_ports(gamestate, setup)
                return self._reach_neutral_frame_zero(
                    gamestate,
                    on_countdown_frame=on_countdown_frame,
                )
            if gamestate.menu_state in (melee.Menu.MAIN_MENU, melee.Menu.PRESS_START):
                self._menu_helper.choose_direct_online(gamestate, self._controller)
            else:
                self._menu_helper.menu_helper_simple(
                    gamestate=gamestate,
                    controller=self._controller,
                    character_selected=setup.character,
                    stage_selected=setup.stage,
                    connect_code=setup.opponent_code,
                    costume=setup.costume,
                    autostart=True,
                )

    def _reach_neutral_frame_zero(
        self,
        gamestate: melee.GameState,
        *,
        on_countdown_frame: Callable[[dict], None] | None = None,
    ) -> dict:
        """Send neutral through the intro and return exact playable frame zero."""
        assert self._console is not None
        assert self._controller is not None
        assert self.ego_port is not None
        while True:
            frame = canonical_frame(gamestate)
            frame_id = frame.get("id")
            if not isinstance(frame_id, int):
                raise RuntimeError(f"netplay countdown frame has invalid id {frame_id!r}")
            if frame_id >= 0:
                if frame_id != 0:
                    raise RuntimeError(f"netplay reached playable frame {frame_id}; expected frame 0")
                pre = frame["ports"][self.ego_port]["leader"]["pre"]
                actual = canonical_pre_to_action(pre)
                if not controller_actions_match(NEUTRAL_CONTROLLER_ACTION, actual):
                    raise RuntimeError(f"local controller is not neutral at frame 0: {actual!r}")
                return frame
            if on_countdown_frame is not None:
                on_countdown_frame(frame)
            self._controller.release_all()
            self._controller.flush()
            gamestate = step_blocking(self._console, self.step_timeout_seconds)
            if gamestate.menu_state not in LIVE_MENU_STATES:
                raise RuntimeError("netplay left the game during the pre-game countdown")

    def _discover_ports(self, gamestate: melee.GameState, setup: NetplaySetup) -> None:
        ports = {port: player for port, player in gamestate.players.items() if port in (1, 2)}
        opponent_matches = [
            port for port, player in ports.items() if getattr(player, "connectCode", "") == setup.opponent_code
        ]
        if len(opponent_matches) == 1:
            self.opponent_port = opponent_matches[0]
            self.ego_port = 3 - self.opponent_port
        else:
            local_matches = [
                port
                for port, player in ports.items()
                if player.character == setup.character and getattr(player, "costume", None) == setup.costume
            ]
            if len(local_matches) == 1:
                self.ego_port = local_matches[0]
                self.opponent_port = 3 - self.ego_port
            elif len(local_matches) > 1:
                raise RuntimeError(
                    f"cannot discover local netplay port: both players use {setup.character.name} costume "
                    f"{setup.costume} and neither player matched the remote connect code"
                )
            else:
                detected = melee.gamestate.port_detector(gamestate, setup.character, setup.costume)
                if detected not in (1, 2):
                    raise RuntimeError(
                        "cannot discover local netplay port: no player matched the remote connect code "
                        f"and port_detector returned {detected!r}"
                    )
                self.ego_port = detected
                self.opponent_port = 3 - detected
        ego = ports.get(self.ego_port)
        if ego is None:
            raise RuntimeError(f"local netplay port {self.ego_port} is absent from the live frame")
        if ego.character != setup.character:
            raise RuntimeError(f"local player selected {ego.character.name}, expected {setup.character.name}")
        logger.info(f"netplay ports: local={self.ego_port}, opponent={self.opponent_port}")

    def step(self, inputs: ControllerInputs | None) -> tuple[dict, bool]:
        """Apply one input and return the next predicted netplay frame."""
        if self._console is None or self._controller is None:
            raise RuntimeError("start_match must complete before step")
        if inputs is None:
            raise TypeError("netplay step requires controller inputs, including for neutral")
        if self._last_frame_id is None:
            raise RuntimeError("start_match did not establish a live frame")
        apply_inputs(self._controller, inputs)
        gamestate = step_blocking(self._console, self.step_timeout_seconds)
        frame = canonical_frame(gamestate)
        in_game = gamestate.menu_state in LIVE_MENU_STATES
        if not in_game:
            self._last_frame_id = None
            return frame, False
        frame_id = frame.get("id")
        if not isinstance(frame_id, int):
            raise RuntimeError(f"netplay frame has invalid id {frame_id!r}")
        if frame_id != self._last_frame_id + 1:
            raise RuntimeError(
                f"netplay returned frame {frame_id} after {self._last_frame_id}; "
                "libmelee rollback filtering did not preserve call alignment"
            )
        self._last_frame_id = frame_id
        return frame, True

    def _kill_dolphin_only(self) -> None:
        kill_dolphin(self._console)

    def _teardown(self) -> None:
        with suppress(Exception):
            atexit.unregister(self._atexit_kill)
        try:
            teardown_console(self._console, self.replay_dir)
        finally:
            logger.info("netplay session closed slippi_port={}", self.slippi_port)
            self._console = None
            self._controller = None
            self._menu_helper = None
            self._last_frame_id = None
            self.ego_port = None
            self.opponent_port = None
