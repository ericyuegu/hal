import concurrent.futures
import platform
import random
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Tuple

import attr
import melee
from loguru import logger
from melee import enums

from hal.constants import ORIGINAL_BUTTONS
from hal.constants import PLAYER_1_PORT
from hal.constants import PLAYER_2_PORT
from hal.constants import Player
from hal.constants import get_opponent
from hal.eval.eval_helper import EpisodeStats
from hal.eval.eval_helper import Matchup
from hal.local_paths import EMULATOR_PATH
from hal.local_paths import EVAL_REPLAY_DIR
from hal.local_paths import ISO_PATH
from hal.training.io import find_latest_idx
from hal.training.io import get_path_friendly_datetime


def _get_console_port(player: Player) -> int:
    return PLAYER_1_PORT if player == "p1" else PLAYER_2_PORT


def find_open_udp_ports(num: int) -> List[int]:
    min_port = 10_000
    max_port = 2**16

    system = platform.system()
    if system == "Linux":
        netstat_command = ["netstat", "-an", "--udp"]
        port_delimiter = ":"
    elif system == "Darwin":
        netstat_command = ["netstat", "-an", "-p", "udp"]
        port_delimiter = "."
    else:
        raise NotImplementedError(f'Unsupported system "{system}"')

    netstat = subprocess.check_output(netstat_command)
    lines = netstat.decode().split("\n")[2:]

    used_ports = set()
    for line in lines:
        words = line.split()
        if not words:
            continue

        address, port = words[3].rsplit(port_delimiter, maxsplit=1)
        if port == "*":
            # TODO: what does this mean? Seems to only happen on Darwin.
            continue

        if address in ("::", "localhost", "0.0.0.0", "*"):
            used_ports.add(int(port))

    available_ports = set(range(min_port, max_port)) - used_ports

    if len(available_ports) < num:
        raise RuntimeError("Not enough available ports.")

    return random.sample(list(available_ports), num)


def get_replay_dir(artifact_dir: Path | None = None, step: int | None = None) -> Path:
    if artifact_dir is None:
        replay_dir = Path(EVAL_REPLAY_DIR) / get_path_friendly_datetime()
    else:
        replay_dir = artifact_dir / "replays"
        step = step or find_latest_idx(artifact_dir)
    if step is not None:
        replay_dir = replay_dir / f"{step:012d}"
    return replay_dir


def get_headless_console_kwargs(
    enable_ffw: bool = True,
    udp_port: int | None = None,
    replay_dir: Path | None = None,
    console_logger: melee.Logger | None = None,
) -> Dict[str, Any]:
    headless_console_kwargs = {
        "gfx_backend": "Null",
        "disable_audio": True,
        "use_exi_inputs": enable_ffw,
        "enable_ffw": enable_ffw,
    }
    emulator_path = EMULATOR_PATH
    if replay_dir is None:
        replay_dir = get_replay_dir()
    replay_dir.mkdir(exist_ok=True, parents=True)
    if udp_port is None:
        udp_port = find_open_udp_ports(1)[0]
    console_kwargs = {
        "path": emulator_path,
        "is_dolphin": True,
        "tmp_home_directory": True,
        "copy_home_directory": False,
        "replay_dir": str(replay_dir),
        "blocking_input": True,
        "slippi_port": udp_port,
        "online_delay": 0,  # 0 frame delay for local evaluation
        "logger": console_logger,
        **headless_console_kwargs,
    }
    return console_kwargs


def get_gui_console_kwargs(
    emulator_path: Path,
    replay_dir: Path,
    udp_port: int | None = None,
    console_logger: melee.Logger | None = None,
) -> Dict[str, Any]:
    """Get console kwargs for GUI-enabled emulator."""
    replay_dir.mkdir(exist_ok=True, parents=True)
    if udp_port is None:
        udp_port = find_open_udp_ports(1)[0]
    console_kwargs = {
        "path": str(emulator_path),
        "is_dolphin": True,
        "tmp_home_directory": True,
        "copy_home_directory": False,
        "replay_dir": str(replay_dir),
        "blocking_input": False,
        "slippi_port": udp_port,
        "online_delay": 0,  # 0 frame delay for local evaluation
        "logger": console_logger,
        "setup_gecko_codes": True,
        "gfx_backend": "",
        "disable_audio": False,
        "use_exi_inputs": False,
        "enable_ffw": False,
    }
    return console_kwargs


@attr.s(auto_attribs=True)
class MatchupMenuHelper:
    controller_1: melee.Controller
    controller_2: melee.Controller
    character_1: melee.Character
    character_2: Optional[melee.Character]
    stage: Optional[melee.Stage]
    opponent_cpu_level: int = 9

    # Internal use
    _player_1_character_selected: bool = False

    def select_character_and_stage(self, gamestate: melee.GameState) -> None:
        """
        Call this helper function every frame to handle menu state logic.

        If character_2 or stage_selected is None, the function will wait for human user.
        """
        if gamestate.menu_state == enums.Menu.MAIN_MENU:
            melee.menuhelper.MenuHelper.choose_versus_mode(gamestate=gamestate, controller=self.controller_1)
        # If we're at the character select screen, choose our character
        elif gamestate.menu_state == enums.Menu.CHARACTER_SELECT:
            melee.menuhelper.MenuHelper.choose_character(
                character=self.character_1,
                gamestate=gamestate,
                controller=self.controller_1,
                cpu_level=0,  # human
                costume=0,
                swag=False,
                start=False,
            )
            if self.character_2 is None:
                return
            melee.menuhelper.MenuHelper.choose_character(
                character=self.character_2,
                gamestate=gamestate,
                controller=self.controller_2,
                cpu_level=self.opponent_cpu_level,
                costume=1,
                swag=False,
                start=True,
            )
        # If we're at the stage select screen, choose a stage
        elif gamestate.menu_state == enums.Menu.STAGE_SELECT:
            if self.stage is None:
                return
            melee.menuhelper.MenuHelper.choose_stage(
                stage=self.stage, gamestate=gamestate, controller=self.controller_1, character=self.character_1
            )
        # If we're at the postgame scores screen, spam START
        elif gamestate.menu_state == enums.Menu.POSTGAME_SCORES:
            melee.menuhelper.MenuHelper.skip_postgame(controller=self.controller_1)


@contextmanager
def console_manager(console: melee.Console, console_logger: melee.Logger | None = None):
    def signal_handler(sig, frame):
        raise KeyboardInterrupt

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        yield
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except TimeoutError:
        raise
    except Exception as e:
        logger.error(
            f"Stopping console due to exception: {e}\nTraceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
        )
        raise
    finally:
        if console_logger is not None:
            console_logger.writelog()
            logger.info("Log file created: " + console_logger.filename)
        signal.signal(signal.SIGINT, original_handler)
        console.stop()
        logger.info("Shutting down cleanly...")


def send_controller_inputs(controller: melee.Controller, inputs: Dict[str, Any]) -> None:
    """
    Press buttons and tilt analog sticks given a dictionary of array-like values (length T for T future time steps).

    Args:
        controller (melee.Controller): Controller object.
        inputs (Dict[str, Any]): Dictionary of controller inputs
    """
    controller.tilt_analog(
        melee.Button.BUTTON_MAIN,
        inputs["main_stick"][0],
        inputs["main_stick"][1],
    )
    controller.tilt_analog(
        melee.Button.BUTTON_C,
        inputs["c_stick"][0],
        inputs["c_stick"][1],
    )
    # Handle shoulder input from either format
    shoulder_value = inputs.get("shoulder", inputs.get("analog_shoulder", 0))
    controller.press_shoulder(
        melee.Button.BUTTON_L,
        shoulder_value,
    )

    buttons_to_press: List[str] = inputs.get("buttons", [])
    for button_str in ORIGINAL_BUTTONS:
        button = getattr(melee.Button, button_str.upper())
        if button_str in buttons_to_press:
            controller.press_button(button)
        else:
            controller.release_button(button)

    controller.flush()


@attr.s(auto_attribs=True)
class EmulatorManager:
    udp_port: int
    player: Player
    replay_dir: Path | None = None
    opponent_cpu_level: int = 9
    matchup: Matchup = Matchup(stage="BATTLEFIELD", ego_character="FOX", opponent_character="FOX")
    episode_stats: EpisodeStats = EpisodeStats()
    max_steps: int = 99999
    latency_warning_threshold: float = 14.0
    console_timeout: float = 5.0
    enable_ffw: bool = True
    debug: bool = False

    def __attrs_post_init__(self) -> None:
        self.console_logger = melee.Logger() if self.debug else None
        console_kwargs = get_headless_console_kwargs(
            enable_ffw=self.enable_ffw,
            udp_port=self.udp_port,
            replay_dir=self.replay_dir,
            console_logger=self.console_logger,
        )
        self.console = melee.Console(**console_kwargs)
        self.ego_controller = melee.Controller(
            console=self.console, port=_get_console_port(self.player), type=melee.ControllerType.STANDARD
        )
        self.opponent_controller = melee.Controller(
            console=self.console, port=_get_console_port(get_opponent(self.player)), type=melee.ControllerType.STANDARD
        )
        self.menu_helper = MatchupMenuHelper(
            controller_1=self.ego_controller,
            controller_2=self.opponent_controller,
            character_1=melee.Character[self.matchup.ego_character],
            character_2=melee.Character[self.matchup.opponent_character],
            stage=melee.Stage[self.matchup.stage],
            opponent_cpu_level=self.opponent_cpu_level,
        )

    def run_game(self) -> Generator[melee.GameState, Tuple[Dict[str, Any], Dict[str, Any] | None], None]:
        """Generator that yields gamestates and receives controller inputs.

        Yields:
            Optional[melee.GameState]: The current game state, or None if the episode is over

        Sends:
            TensorDict: Controller inputs to be applied to the game
        """
        # Run the console
        self.console.run(iso_path=ISO_PATH)  # Do not pass dolphin_user_path to avoid overwriting init kwargs
        # Connect to the console
        logger.debug("Connecting to console...")
        if not self.console.connect():
            logger.debug("ERROR: Failed to connect to the console.")
            sys.exit(-1)
        logger.debug("Console connected")

        # Plug our controller in
        #   Due to how named pipes work, this has to come AFTER running dolphin
        #   NOTE: If you're loading a movie file, don't connect the controller,
        #   dolphin will hang waiting for input and never receive it
        logger.debug("Connecting controller 1 to console...")
        if not self.ego_controller.connect():
            logger.debug("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        logger.debug("Controller 1 connected")
        logger.debug("Connecting controller 2 to console...")
        if not self.opponent_controller.connect():
            logger.debug("ERROR: Failed to connect the controller.")
            sys.exit(-1)
        logger.debug("Controller 2 connected")

        i = 0
        match_started = False

        # Wrap console manager inside a thread for timeouts
        # Important that console manager context goes second to gracefully handle keyboard interrupts, timeouts, and all other exceptions
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor, console_manager(
            console=self.console, console_logger=self.console_logger
        ):
            logger.debug(
                f"Starting episode on {self.matchup.stage}: {self.matchup.ego_character} vs. {self.matchup.opponent_character}"
            )
            while i < self.max_steps:
                # Wrap `console.step()` in a thread with timeout
                future = executor.submit(self.console.step)
                try:
                    step_start = time.perf_counter()
                    gamestate = future.result(timeout=self.console_timeout)
                    step_time = time.perf_counter() - step_start
                except concurrent.futures.TimeoutError:
                    logger.error("console.step() timed out")
                    raise

                if gamestate is None:
                    logger.info("Gamestate is None")
                    continue

                if self.console.processingtime * 1000 > self.latency_warning_threshold:
                    logger.debug("Last frame took " + str(self.console.processingtime * 1000) + "ms to process.")

                if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                    if match_started:
                        logger.debug("Match ended")
                        break
                    self.menu_helper.select_character_and_stage(gamestate)
                else:
                    if not match_started:
                        match_started = True
                        logger.debug("Match started")

                    # Yield gamestate and receive controller inputs
                    controller_inputs = yield gamestate
                    if controller_inputs is None:
                        logger.error("Controller inputs are None")
                    else:
                        ego_controller_inputs, opponent_controller_inputs = controller_inputs
                        send_start = time.perf_counter()
                        send_controller_inputs(self.ego_controller, ego_controller_inputs)
                        if opponent_controller_inputs is not None:
                            send_controller_inputs(self.opponent_controller, opponent_controller_inputs)
                        send_time = time.perf_counter() - send_start

                        if i % 60 == 0:
                            logger.debug(
                                f"Console.step() time: {step_time*1000:.2f}ms, controller send time: {send_time*1000:.2f}ms"
                            )

                    self.episode_stats.update(gamestate)
                    if self.console_logger is not None:
                        self.console_logger.writeframe()
                    i += 1
