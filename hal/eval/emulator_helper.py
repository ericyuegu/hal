import platform
import random
import signal
import subprocess
from concurrent.futures import TimeoutError
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import melee
from loguru import logger
from melee import enums
from melee.menuhelper import MenuHelper

from hal.eval.emulator_paths import REMOTE_EMULATOR_PATH
from hal.eval.emulator_paths import REMOTE_EVAL_REPLAY_DIR
from hal.training.io import ARTIFACT_DIR_ROOT
from hal.training.io import get_path_friendly_datetime
from hal.training.utils import get_git_repo_root


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
def console_manager(console: melee.Console, shutdown_timeout: float = 5.0):
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
        signal.signal(signal.SIGINT, original_handler)
        console.stop()
        logger.info("Shutting down cleanly...")
