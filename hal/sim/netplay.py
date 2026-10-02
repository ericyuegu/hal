"""One local Dolphin session for Slippi netplay."""

import atexit
import hashlib
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal
from typing import Protocol
from typing import Self

import melee
import melee.console
from loguru import logger

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import apply_inputs
from hal.sim.session import LIVE_MENU_STATES
from hal.sim.session import DolphinGraphicsBackend
from hal.sim.session import FrameTimeout
from hal.sim.session import canonical_frame
from hal.sim.session import fix_dolphin_ini_case
from hal.sim.session import kill_dolphin
from hal.sim.session import known_dolphin_version
from hal.sim.session import popen_with_pdeathsig
from hal.sim.session import set_dolphin_stream_output
from hal.sim.session import step_blocking
from hal.sim.session import teardown_console

_SLIPPI_3_6_4_LINUX_SHA256 = "e0f984e5bbecb98e3a746da1f173a475b06c3a1ba6b73e2e31bbe85a5f5a5e8a"


def tested_dolphin_version(dolphin_path: str) -> melee.console.DolphinVersion:
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


class CountdownEnded(RuntimeError):
    """The remote game returned to a menu before playable frame zero."""


class ConnectAbandoned(Exception):
    """The caller gave up on the remote player before the match went live."""


class PlayerDisconnected(Exception):
    """The remote player left the direct-mode session."""


class PlayerIdle(Exception):
    """The remote player stayed at character select past the idle timeout."""


class PlayerNoShow(Exception):
    """The remote player did not connect within the connect timeout."""


def _never_abandoned() -> bool:
    return False


class MenuDriver(Protocol):
    def __call__(self, state: melee.GameState | None, controller: melee.Controller) -> bool:
        """Return true when inputs changed and need one flush.

        A missing state lets the driver release a timed button press while a
        Slippi scene does not emit menu observations. Never called in a game.
        """
        ...


@dataclass(frozen=True, slots=True)
class DirectSelection:
    revision: int
    character: melee.Character
    costume: int
    stage: melee.Stage
    identity: str


# A match start also reports the code-entry submenu, for about a second; a
# disconnect leaves Slippi on code entry, so only a held code-entry screen counts.
_DISCONNECT_CONFIRM_SECONDS = 2.0


class DirectMenuDriver:
    """Drive Slippi 3.6.4 direct mode for one Dolphin session.

    Slippi has no un-ready, and the bot cannot see the remote lock-in, so the bot
    hovers its character and locks in last. Start while connected and locked is a
    no-op; once the remote player has gone, Start opens code entry, which is the
    only observable sign of a disconnect.
    """

    def __init__(
        self,
        *,
        opponent_code: str,
        selection: Callable[[], DirectSelection],
        lock_requests: Callable[[], int],
        connect_timeout_seconds: float,
        idle_timeout_seconds: float,
        hold_seconds: float,
        hold_cap_seconds: float,
        probe_interval_seconds: float,
        on_change: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_change = on_change
        self._opponent_code = opponent_code
        self._selection = selection
        self._lock_requests = lock_requests
        self._connect_timeout = connect_timeout_seconds
        self._idle_timeout = idle_timeout_seconds
        self._hold = hold_seconds
        self._hold_cap = hold_cap_seconds
        self._probe_interval = probe_interval_seconds
        self._clock = clock
        # Slippi keeps the Frozen Stadium toggle for the life of the Dolphin process.
        self._frozen_stadium = False
        self.connected = False
        self.phase: Literal["waiting_for_player", "character_select"] = "waiting_for_player"
        self.deadline: float | None = None
        self.locked: DirectSelection | None = None
        self._helper = melee.MenuHelper()
        self._arrived: float | None = None
        self._hold_until = 0.0
        self._hovered: DirectSelection | None = None
        self._lock_seen = 0
        self._next_probe = 0.0
        self._connect_started: float | None = None
        self._code_entry_since: float | None = None

    def _update(
        self,
        phase: Literal["waiting_for_player", "character_select"],
        deadline: float | None,
        locked: DirectSelection | None,
    ) -> None:
        before = (self.phase, self.deadline, self.locked)
        self.phase = phase
        self.deadline = deadline
        self.locked = locked
        if (self.phase, self.deadline, self.locked) != before:
            self._on_change()

    def begin_game(self) -> None:
        self._helper = melee.MenuHelper()
        self._helper.frozen_stadium_selected = self._frozen_stadium
        self._arrived = None
        self._hovered = None
        self._code_entry_since = None
        # Only a request made during this character select may skip the hold.
        self._lock_seen = self._lock_requests()
        phase: Literal["waiting_for_player", "character_select"] = (
            "character_select" if self.connected else "waiting_for_player"
        )
        self._update(phase, None, None)

    def __call__(self, state: melee.GameState | None, controller: melee.Controller) -> bool:
        if state is None:
            # Polls between frames must not flush: a release here would cancel the
            # last press before the game samples input.
            return False
        now = self._clock()
        menu, submenu = state.menu_state, getattr(state, "submenu", None)
        if menu in (melee.Menu.MAIN_MENU, melee.Menu.PRESS_START):
            if self.connected:
                raise PlayerDisconnected("Slippi returned to the main menu")
            melee.MenuHelper.choose_direct_online(state, controller)
            return True
        if menu == melee.Menu.POSTGAME_SCORES:
            self._helper.skip_postgame(controller)
            return True
        if menu == melee.Menu.STAGE_SELECT:
            selection = self.locked or self._selection()
            self._helper.choose_stage(
                stage=selection.stage,
                gamestate=state,
                controller=controller,
                character=selection.character,
                frozen_stadium=True,
                autostart=True,
            )
            self._frozen_stadium = self._helper.frozen_stadium_selected
            return True
        if menu != melee.Menu.SLIPPI_ONLINE_CSS:
            controller.release_all()
            return True
        if submenu == melee.SubMenu.NAME_ENTRY_SUBMENU:
            if state.ready_to_start:
                # The ready-to-fight banner: both players locked in and the match is starting.
                self._code_entry_since = None
                controller.release_all()
                return True
            if self.connected:
                if self._code_entry_since is None:
                    self._code_entry_since = now
                if now - self._code_entry_since >= _DISCONNECT_CONFIRM_SECONDS:
                    raise PlayerDisconnected("Slippi stayed on code entry")
                controller.release_all()
                return True
            if self._connect_started is None:
                self._connect_started = now
                self._update("waiting_for_player", now + self._connect_timeout, self.locked)
            self._helper.enter_direct_code(
                gamestate=state,
                controller=controller,
                connect_code=self._opponent_code,
            )
            return True
        self._code_entry_since = None
        if not self.connected:
            return self._connect(state, controller, now)
        return self._between_games(state, controller, now)

    def _connect(self, state: melee.GameState, controller: melee.Controller, now: float) -> bool:
        # Game 1: pick the newest selection, then Start opens code entry; the match
        # starts when both players have entered each other's codes.
        if self._connect_started is not None and now >= self._connect_started + self._connect_timeout:
            raise PlayerNoShow(f"no connection within {self._connect_timeout:.0f}s")
        selection = self._selection()
        self._update(self.phase, self.deadline, selection)
        self._helper.choose_character(
            character=selection.character,
            gamestate=state,
            controller=controller,
            costume=selection.costume,
            start=True,
        )
        return True

    def _between_games(self, state: melee.GameState, controller: melee.Controller, now: float) -> bool:
        if self._arrived is None:
            self._arrived = now
        if now >= self._arrived + self._idle_timeout:
            raise PlayerIdle(f"no game started within {self._idle_timeout:.0f}s")
        if self.locked is not None:
            self._update("character_select", self._arrived + self._idle_timeout, self.locked)
            if now >= self._next_probe:
                self._next_probe = now + self._probe_interval
                controller.release_all()
                controller.press_button(melee.Button.BUTTON_START)
            else:
                controller.release_button(melee.Button.BUTTON_START)
            return True
        selection = self._selection()
        if self._hovered is None or (
            selection.character,
            selection.costume,
            selection.stage,
            selection.identity,
        ) != (
            self._hovered.character,
            self._hovered.costume,
            self._hovered.stage,
            self._hovered.identity,
        ):
            self._hold_until = min(now + self._hold, self._arrived + self._hold_cap)
        self._hovered = selection
        requested = self._lock_requests() > self._lock_seen
        hovering = state.players[1].character == selection.character
        if hovering and (requested or now >= self._hold_until):
            self._lock_seen = self._lock_requests()
            self._next_probe = now + self._probe_interval
            controller.release_all()
            controller.press_button(melee.Button.BUTTON_START)
            self._update("character_select", self._hold_until, selection)
            return True
        self._update("character_select", self._hold_until, None)
        self._helper.choose_character(
            character=selection.character,
            gamestate=state,
            controller=controller,
            costume=selection.costume,
            start=False,
        )
        return True


@dataclass(frozen=True, slots=True)
class NetplaySetup:
    """Match choices; local_code identifies our port against unknown opponents."""

    character: melee.Character
    opponent_code: str
    costume: int = 0
    stage: melee.Stage = melee.Stage.FINAL_DESTINATION
    local_code: str | None = None


class NetplaySession:
    """Drive a Dolphin netplay session with explicit real-time I/O."""

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
        connect_abandoned: Callable[[], bool] = _never_abandoned,
        realtime: bool = False,
        graphics_backend: DolphinGraphicsBackend = "Vulkan",
        stream_output: bool = False,
        menu_driver: MenuDriver | None = None,
    ) -> None:
        if online_delay not in (2, 3):
            raise ValueError(f"online_delay must be 2 or 3, got {online_delay}")
        if graphics_backend not in ("Vulkan", "OGL"):
            raise ValueError(f"unsupported Dolphin graphics backend {graphics_backend!r}")
        if type(stream_output) is not bool:
            raise TypeError("stream_output must be a boolean")
        if menu_driver is not None and not realtime:
            raise ValueError("a custom menu driver requires real-time netplay")
        self.menu_driver = menu_driver
        self.frame_times: list[float] = []
        self.realtime = realtime
        self.iso_path = str(iso_path)
        self.dolphin_path = str(dolphin_path)
        self.user_json_path = str(user_json_path)
        self.online_delay = online_delay
        self.replay_dir = str(Path(replay_dir).resolve())
        self.slippi_port = slippi_port
        self.step_timeout_seconds = step_timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.connect_abandoned = connect_abandoned
        self.graphics_backend = graphics_backend
        self.stream_output = stream_output
        self.ego_port: int | None = None
        self.opponent_port: int | None = None
        self.ego_character: melee.Character | None = None
        self._console: melee.Console | None = None
        self._controller: melee.Controller | None = None
        self._menu_helper: melee.MenuHelper | None = None
        self._last_frame_id: int | None = None
        self._atexit_kill = self._kill_dolphin_only

    def __enter__(self) -> Self:
        version = tested_dolphin_version(self.dolphin_path)
        with known_dolphin_version(version):
            self._console = melee.Console(
                path=self.dolphin_path,
                slippi_port=self.slippi_port,
                online_delay=self.online_delay,
                user_json_path=self.user_json_path,
                blocking_input=not self.realtime,
                polling_mode=True,
                # Console.step() flushes every controller before it polls. A
                # zero-timeout polling loop can therefore send several FLUSH
                # commands for one returned frame and destroy call alignment.
                polling_timeout=0.0 if self.realtime else self.step_timeout_seconds,
                skip_rollback_frames=True,
                rollback_resolution="first",
                gfx_backend=self.graphics_backend,
                fullscreen=False,
                disable_audio=not self.stream_output,
                audio_backend="Pulse" if self.stream_output else "",
                tmp_home_directory=True,
                save_replays=True,
                replay_dir=self.replay_dir,
                replay_monthly_folders=False,
                emulation_speed=1.0 if self.realtime else 0,
                use_exi_inputs=False,
                enable_ffw=False,
            )
        fix_dolphin_ini_case(self._console)
        set_dolphin_stream_output(self._console, self.stream_output)
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
        on_countdown_frame: Callable[[dict], ControllerInputs] | None = None,
        on_countdown_observation: Callable[[dict], None] | None = None,
    ) -> dict:
        """Connect and emit countdown states before returning frame zero."""
        if self._console is None:
            raise RuntimeError("NetplaySession must be used as a context manager")
        if self._controller is not None:
            raise RuntimeError("NetplaySession has already launched Dolphin")
        if not setup.opponent_code and not (setup.local_code and self.menu_driver is not None):
            raise ValueError("opponent_code must be non-empty, or a local code and menu driver are required")
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
        # Batch mode suppresses Dolphin's launcher window. The render window is
        # still visible, so a stream display contains only gameplay.
        with popen_with_pdeathsig(dolphin_batch=True):
            self._console.run(iso_path=self.iso_path)
        if not self._controller.connect():
            raise RuntimeError("failed to connect the local controller")
        self._controller.release_all()
        self._controller.flush()
        if not self._console.connect():
            raise RuntimeError("failed to connect to Dolphin Slippi server")
        self._menu_helper = melee.MenuHelper()
        logger.info("waiting for netplay match; remote code={}", setup.opponent_code or "matchmaking")
        first_frame = self._navigate_to_live(
            setup, on_countdown_frame=on_countdown_frame, on_countdown_observation=on_countdown_observation
        )
        frame_id = first_frame.get("id")
        if not isinstance(frame_id, int):
            raise RuntimeError(f"first netplay frame has invalid id {frame_id!r}")
        self._last_frame_id = frame_id
        return first_frame

    def start_rematch(
        self,
        setup: NetplaySetup,
        *,
        on_countdown_frame: Callable[[dict], ControllerInputs] | None = None,
        on_countdown_observation: Callable[[dict], None] | None = None,
    ) -> dict:
        """Navigate to a rematch and emit countdown states before frame zero."""
        if self._console is None or self._controller is None or self._menu_helper is None:
            raise RuntimeError("start_match must complete before start_rematch")
        if self._last_frame_id is not None:
            raise RuntimeError("the current netplay match has not ended")
        # MenuHelper latches its choices, so a rematch needs a fresh one, but Slippi
        # keeps the Frozen Stadium toggle for the whole Dolphin session.
        frozen = self._menu_helper.frozen_stadium_selected
        self._menu_helper = melee.MenuHelper()
        self._menu_helper.frozen_stadium_selected = frozen
        self._controller.release_all()
        self._controller.flush()
        first_frame = self._navigate_to_live(
            setup, on_countdown_frame=on_countdown_frame, on_countdown_observation=on_countdown_observation
        )
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
        gamestate = self._read_state()
        if gamestate.menu_state in LIVE_MENU_STATES:
            raise RuntimeError("netplay entered a game while waiting for a rematch")
        return gamestate.menu_state

    def _navigate_to_live(
        self,
        setup: NetplaySetup,
        *,
        on_countdown_frame: Callable[[dict], ControllerInputs] | None = None,
        on_countdown_observation: Callable[[dict], None] | None = None,
    ) -> dict:
        assert self._console is not None
        assert self._controller is not None
        assert self._menu_helper is not None
        deadline = time.monotonic() + self.connect_timeout_seconds
        last_status: tuple[melee.Menu, object] | None = None
        while True:
            self._raise_if_connect_abandoned()
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"did not reach IN_GAME within {self.connect_timeout_seconds:.0f}s "
                    "while waiting for the remote player"
                )
            if self.menu_driver is None:
                gamestate = self._read_state(during_connect=True)
            else:
                gamestate = self._console.step(flush_controllers=False)
                self._raise_if_connect_abandoned()
                if gamestate is None:
                    self._raise_if_dolphin_exited()
                    if self.menu_driver(None, self._controller):
                        self._controller.flush()
                    time.sleep(0.0005)
                    continue
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
                return self._reach_frame_zero(
                    gamestate,
                    on_countdown_frame=on_countdown_frame,
                    on_countdown_observation=on_countdown_observation,
                )
            if self.menu_driver is not None:
                if self.menu_driver(gamestate, self._controller):
                    self._controller.flush()
            elif gamestate.menu_state in (melee.Menu.MAIN_MENU, melee.Menu.PRESS_START):
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

    def _reach_frame_zero(
        self,
        gamestate: melee.GameState,
        *,
        on_countdown_frame: Callable[[dict], ControllerInputs] | None = None,
        on_countdown_observation: Callable[[dict], None] | None = None,
    ) -> dict:
        """Apply countdown inputs and return exact playable frame zero."""
        assert self._console is not None
        assert self._controller is not None
        assert self.ego_port is not None
        while True:
            self._raise_if_connect_abandoned()
            frame = canonical_frame(gamestate)
            frame_id = frame.get("id")
            if not isinstance(frame_id, int):
                raise RuntimeError(f"netplay countdown frame has invalid id {frame_id!r}")
            if frame_id >= 0:
                if frame_id != 0:
                    raise RuntimeError(f"netplay reached playable frame {frame_id}; expected frame 0")
                return frame
            if on_countdown_observation is not None:
                on_countdown_observation(frame)
            self._raise_if_connect_abandoned()
            if self.realtime:
                next_state = self._console.step(flush_controllers=False)
                self._raise_if_connect_abandoned()
                if next_state is not None:
                    gamestate = next_state
                    if gamestate.menu_state not in LIVE_MENU_STATES:
                        raise CountdownEnded("netplay left the game during the pre-game countdown")
                    continue
            inputs = NEUTRAL_CONTROLLER_ACTION if on_countdown_frame is None else on_countdown_frame(frame)
            self._raise_if_connect_abandoned()
            apply_inputs(self._controller, inputs)
            gamestate = self._read_state(during_connect=True)
            if gamestate.menu_state not in LIVE_MENU_STATES:
                raise CountdownEnded("netplay left the game during the pre-game countdown")

    def _discover_ports(self, gamestate: melee.GameState, setup: NetplaySetup) -> None:
        # libmelee 0.47.0 can key pre-frame players with NumPy integer scalars.
        ports = {int(port): player for port, player in gamestate.players.items() if port in (1, 2)}
        opponent_matches = [
            port for port, player in ports.items() if getattr(player, "connectCode", "") == setup.opponent_code
        ]
        if setup.local_code is not None:
            local_matches = [
                port for port, player in ports.items() if getattr(player, "connectCode", "") == setup.local_code
            ]
            if len(local_matches) != 1:
                raise RuntimeError(
                    "cannot discover local netplay port: local connect code must match exactly one player"
                )
            self.ego_port = local_matches[0]
            self.opponent_port = 3 - self.ego_port
        elif len(opponent_matches) == 1:
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
                self.ego_port = int(detected)
                self.opponent_port = 3 - self.ego_port
        ego = ports.get(self.ego_port)
        if ego is None:
            raise RuntimeError(f"local netplay port {self.ego_port} is absent from the live frame")
        self.ego_character = ego.character
        # A custom driver owns the selection and its caller checks the locked character.
        if self.menu_driver is None and ego.character != setup.character:
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
        gamestate = self._read_state()
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

    def _raise_if_connect_abandoned(self) -> None:
        if self.connect_abandoned():
            raise ConnectAbandoned("stopped waiting for the remote player")

    def _raise_if_dolphin_exited(self) -> None:
        if self._console is None:
            return
        process = self._console._process
        if process is not None and process.poll() is not None:
            raise FrameTimeout("Dolphin exited while waiting for a frame")

    def _read_state(self, *, during_connect: bool = False) -> melee.GameState:
        if self._console is None:
            raise RuntimeError("netplay console is not initialized")
        if during_connect:
            self._raise_if_connect_abandoned()
        if not self.realtime:
            state = step_blocking(self._console, self.step_timeout_seconds)
            if during_connect:
                self._raise_if_connect_abandoned()
            return state
        if self._controller is not None:
            self._controller.flush()
        timeout_seconds = self.connect_timeout_seconds if during_connect else self.step_timeout_seconds
        deadline = time.monotonic() + timeout_seconds
        while True:
            if during_connect:
                self._raise_if_connect_abandoned()
            state = self._console.step(flush_controllers=False)
            if during_connect:
                self._raise_if_connect_abandoned()
            if state is not None:
                return state
            self._raise_if_dolphin_exited()
            if time.monotonic() >= deadline:
                raise FrameTimeout("netplay observation stream stalled")
            time.sleep(0.0005)

    def submit(self, inputs: ControllerInputs) -> None:
        """Flush exactly the state selected by the game-frame scheduler."""
        if self._controller is None:
            raise RuntimeError("netplay controller is not connected")
        apply_inputs(self._controller, inputs)
        self._controller.flush()

    def read_frames(self, timeout_seconds: float | None = None) -> tuple[list[dict], bool]:
        """Wait for one observation, then drain available observations without writes."""
        if self._console is None or not self.realtime or self._last_frame_id is None:
            raise RuntimeError("read_frames requires an active real-time match")
        deadline = time.monotonic() + (timeout_seconds if timeout_seconds is not None else self.step_timeout_seconds)
        frames = []
        self.frame_times = []
        while True:
            state = self._console.step(flush_controllers=False)
            if state is None:
                if frames:
                    return frames, True
                if time.monotonic() >= deadline:
                    raise FrameTimeout("netplay observation stream stalled")
                time.sleep(0.0005)
                continue
            self.frame_times.append(time.perf_counter())
            frame = canonical_frame(state)
            if state.menu_state not in LIVE_MENU_STATES:
                self._last_frame_id = None
                frames.append(frame)
                return frames, False
            frame_id = frame.get("id")
            if not isinstance(frame_id, int):
                raise RuntimeError("invalid netplay frame ID")
            if self._last_frame_id is not None and frame_id <= self._last_frame_id:
                self.frame_times.pop()
                continue
            self._last_frame_id = frame_id
            frames.append(frame)

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
