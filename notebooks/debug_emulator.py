import sys
from typing import Generator
from typing import Optional

import melee
import torch.multiprocessing as mp
from loguru import logger
from tensordict import TensorDict

from hal.eval.emulator_helper import console_manager
from hal.eval.emulator_helper import get_console_kwargs
from hal.eval.emulator_helper import self_play_menu_helper
from hal.eval.emulator_paths import REMOTE_CISO_PATH
from hal.eval.emulator_paths import REMOTE_DOLPHIN_HOME_PATH

PLAYER_1_PORT = 1
PLAYER_2_PORT = 2


def run_episode(max_steps: int = 8 * 60 * 60) -> Generator[Optional[melee.GameState], TensorDict, None]:
    console_kwargs = get_console_kwargs()
    console = melee.Console(**console_kwargs)

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


def run_episode_wrapper():
    for _ in run_episode():
        pass


if __name__ == "__main__":
    cpu_processes = []
    for i in range(2):
        p: mp.Process = mp.Process(
            target=run_episode_wrapper,
        )
        p.start()
        cpu_processes.append(p)

    for p in cpu_processes:
        p.join()
