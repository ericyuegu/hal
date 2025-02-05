import concurrent.futures
import platform
import random
import signal
import subprocess
import sys
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import List

import attr
import melee
from constants import Player
from constants import get_opponent
from loguru import logger
from melee import enums
from melee.menuhelper import MenuHelper
from tensordict import TensorDict

from hal.constants import PLAYER_1_PORT
from hal.constants import PLAYER_2_PORT
from hal.emulator_paths import REMOTE_CISO_PATH
from hal.emulator_paths import REMOTE_EMULATOR_PATH
from hal.emulator_paths import REMOTE_EVAL_REPLAY_DIR
from hal.eval.eval_helper import EpisodeStats
from hal.eval.eval_helper import send_controller_inputs
from hal.training.io import ARTIFACT_DIR_ROOT
from hal.training.io import get_path_friendly_datetime
from hal.training.utils import get_git_repo_root


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
        replay_dir = Path(REMOTE_EVAL_REPLAY_DIR) / get_path_friendly_datetime()
    else:
        replay_dir = Path(REMOTE_EVAL_REPLAY_DIR) / artifact_dir.relative_to(get_git_repo_root() / ARTIFACT_DIR_ROOT)
    if step is not None:
        replay_dir = replay_dir / f"{step:012d}"
    return replay_dir


def get_console_kwargs(
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
    emulator_path = REMOTE_EMULATOR_PATH
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


def self_play_menu_helper(
    gamestate: melee.GameState,
    controller_1: melee.Controller,
    controller_2: melee.Controller,
    character_1: melee.Character,
    character_2: melee.Character,
    stage_selected: melee.Stage,
    opponent_cpu_level: int = 9,
) -> None:
    if gamestate.menu_state == enums.Menu.MAIN_MENU:
        MenuHelper.choose_versus_mode(gamestate=gamestate, controller=controller_1)
    # If we're at the character select screen, choose our character
    elif gamestate.menu_state == enums.Menu.CHARACTER_SELECT:
        player_1 = gamestate.players[controller_1.port]
        player_1_character_selected = player_1.character == character_1

        if not player_1_character_selected:
            MenuHelper.choose_character(
                character=character_1,
                gamestate=gamestate,
                controller=controller_1,
                cpu_level=0,
                costume=0,
                swag=False,
                start=False,
            )
        else:
            MenuHelper.choose_character(
                character=character_2,
                gamestate=gamestate,
                controller=controller_2,
                cpu_level=opponent_cpu_level,
                costume=1,
                swag=False,
                start=True,
            )
    # If we're at the stage select screen, choose a stage
    elif gamestate.menu_state == enums.Menu.STAGE_SELECT:
        MenuHelper.choose_stage(
            stage=stage_selected, gamestate=gamestate, controller=controller_1, character=character_1
        )
    # If we're at the postgame scores screen, spam START
    elif gamestate.menu_state == enums.Menu.POSTGAME_SCORES:
        MenuHelper.skip_postgame(controller=controller_1)


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
        pass
    except Exception as e:
        logger.error(f"Stopping console due to exception: {e}")
    finally:
        if console_logger is not None:
            console_logger.writelog()
            logger.info("Log file created: " + console_logger.filename)
        signal.signal(signal.SIGINT, original_handler)
        console.stop()
        logger.info("Shutting down cleanly...")


@attr.s(auto_attribs=True)
class EmulatorManager:
    udp_port: int
    player: Player
    opponent_cpu_level: int = 9
    replay_dir: Path | None = None
    episode_stats: EpisodeStats = EpisodeStats()
    max_steps: int = 99999
    latency_warning_threshold: float = 14.0
    console_timeout: float = 2.0
    enable_ffw: bool = True
    debug: bool = False

    def __attrs_post_init__(self) -> None:
        self.console_logger = melee.Logger() if self.debug else None
        console_kwargs = get_console_kwargs(
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
        logger.info(f"Emu manager initialized for {attr.asdict(self)}")

    def run_game(self) -> Generator[melee.GameState, TensorDict, None]:
        """Generator that yields gamestates and receives controller inputs.

        Yields:
            Optional[melee.GameState]: The current game state, or None if the episode is over

        Sends:
            TensorDict: Controller inputs to be applied to the game
        """
        # Run the console
        self.console.run(iso_path=REMOTE_CISO_PATH)  # Do not pass dolphin_user_path to avoid overwriting init kwargs
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
            logger.debug("Starting episode")
            while i < self.max_steps:
                # Wrap `console.step()` in a thread with timeout
                future = executor.submit(self.console.step)
                try:
                    gamestate = future.result(timeout=self.console_timeout)
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

                    self_play_menu_helper(
                        gamestate=gamestate,
                        controller_1=self.ego_controller,
                        controller_2=self.opponent_controller,
                        # TODO: select characters programmatically
                        character_1=melee.Character.FOX,
                        character_2=melee.Character.FOX,
                        stage_selected=melee.Stage.BATTLEFIELD,
                        opponent_cpu_level=self.opponent_cpu_level,
                    )
                else:
                    if not match_started:
                        match_started = True
                        logger.debug("Match started")

                    # Yield gamestate and receive controller inputs
                    # logger.debug(f"Yielding gamestate {i}")
                    controller_inputs = yield gamestate
                    # logger.debug(f"Controller inputs: {controller_inputs}")
                    if controller_inputs is None:
                        logger.error("Controller inputs are None")
                    else:
                        # logger.debug("Sending controller inputs")
                        send_controller_inputs(self.ego_controller, controller_inputs)

                    self.episode_stats.update(gamestate)
                    if self.console_logger is not None:
                        self.console_logger.writeframe()
                    i += 1
