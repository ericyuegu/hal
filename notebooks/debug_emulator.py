import platform
import random
import subprocess
import sys
from typing import Generator
from typing import Optional

import melee
from loguru import logger
from tensordict import TensorDict

from hal.eval.emulator_helper import console_manager
from hal.eval.emulator_helper import get_console_kwargs
from hal.eval.emulator_helper import self_play_menu_helper
from hal.eval.emulator_paths import REMOTE_CISO_PATH
from hal.eval.emulator_paths import REMOTE_DOLPHIN_HOME_PATH

PLAYER_1_PORT = 1
PLAYER_2_PORT = 2


def find_open_udp_ports(num: int):
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


def run_episode(
    rank: int, port: int, max_steps: int = 8 * 60 * 60
) -> Generator[Optional[melee.GameState], TensorDict, None]:
    console_kwargs = get_console_kwargs()
    console = melee.Console(**console_kwargs, slippi_port=port)
    logger.info(f"Worker {rank}: slippi address {console.slippi_address}, port {console.slippi_port}")

    controller_1 = melee.Controller(console=console, port=PLAYER_1_PORT, type=melee.ControllerType.STANDARD)
    controller_2 = melee.Controller(console=console, port=PLAYER_2_PORT, type=melee.ControllerType.STANDARD)

    # Run the console
    console.run(iso_path=REMOTE_CISO_PATH, dolphin_user_path=REMOTE_DOLPHIN_HOME_PATH)
    # Connect to the console
    logger.info("Connecting to console...")
    if not console.connect():
        logger.info("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    logger.info("Console connected")

    # Plug our controller in
    #   Due to how named pipes work, this has to come AFTER running dolphin
    #   NOTE: If you're loading a movie file, don't connect the controller,
    #   dolphin will hang waiting for input and never receive it
    logger.info("Connecting controller 1 to console...")
    if not controller_1.connect():
        logger.info("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    logger.info("Controller 1 connected")
    logger.info("Connecting controller 2 to console...")
    if not controller_2.connect():
        logger.info("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    logger.info("Controller 2 connected")

    i = 0
    match_started = False
    with console_manager(console):
        logger.info("Starting episode")
        try:
            while i < max_steps:
                gamestate = console.step()
                if gamestate is None:
                    logger.info("Gamestate is None")
                    break
                logger.info(f"Iteration {i}: Menu state: {gamestate.menu_state}")

                if console.processingtime * 1000 > 12:
                    logger.info("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")

                if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                    logger.info("Menu helper")
                    if match_started:
                        break

                    self_play_menu_helper(
                        gamestate=gamestate,
                        controller_1=controller_1,
                        controller_2=controller_2,
                        character_1=melee.Character.FOX,
                        character_2=melee.Character.FOX,
                        stage_selected=melee.Stage.BATTLEFIELD,
                    )
                else:
                    if not match_started:
                        match_started = True
                        logger.info("Match started")

                    yield gamestate

                    i += 1
        finally:
            # Signal end of episode
            yield None


def run_episode_wrapper(rank: int, port: int) -> None:
    for _ in run_episode(rank=rank, port=port):
        pass


# if __name__ == "__main__":
#     cpu_processes = []
#     ports = find_open_udp_ports(2)
#     for i, port in enumerate(ports):
#         p: mp.Process = mp.Process(
#             target=run_episode_wrapper,
#             kwargs=dict(rank=i, port=port),
#         )
#         p.start()
#         cpu_processes.append(p)

#     for p in cpu_processes:
#         p.join()

if __name__ == "__main__":
    for i, gamestate in enumerate(run_episode(rank=0, port=51441)):
        if i % 60 == 0:
            print(f"Iteration {i}: {gamestate}")
