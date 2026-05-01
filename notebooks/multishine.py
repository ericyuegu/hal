#!/usr/bin/python3
"""This example program demonstrates how to use the Melee API to run a console,
setup controllers, and send button presses over to a console."""

import argparse
import concurrent.futures
import random
import signal
import sys
from pathlib import Path

import melee
from loguru import logger

from hal.emulator_helper import MatchupMenuHelper
from hal.emulator_helper import console_manager
from hal.emulator_helper import get_gui_console_kwargs
from hal.local_paths import EMULATOR_PATH
from hal.local_paths import EVAL_REPLAY_DIR
from hal.local_paths import ISO_PATH


def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError(f"{value} is an invalid controller port. Must be 1, 2, 3, or 4.")
    return ivalue


def main():
    parser = argparse.ArgumentParser(description="Example of libmelee in action")
    parser.add_argument(
        "--port", "-p", type=check_port, help="The controller port (1-4) your AI will play on", default=2
    )
    parser.add_argument(
        "--opponent", "-o", type=check_port, help="The controller port (1-4) the opponent will play on", default=1
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode. Creates a CSV of all game states")
    parser.add_argument("--connect_code", "-t", default="", help="Direct connect code to connect to in Slippi Online")

    args = parser.parse_args()

    # This logger object is useful for retroactively debugging issues in your bot
    #   You can write things to it each frame, and it will create a CSV file describing the match
    log = None
    if args.debug:
        log = melee.Logger()

    # Create our Console object.
    #   This will be one of the primary objects that we will interface with.
    #   The Console represents the virtual or hardware system Melee is playing on.
    #   Through this object, we can get "GameState" objects per-frame so that your
    #       bot can actually "see" what's happening in the game
    console_kwargs = get_gui_console_kwargs(emulator_path=Path(EMULATOR_PATH), replay_dir=Path(EVAL_REPLAY_DIR))
    logger.info(f"Console kwargs: {console_kwargs}")
    console = melee.Console(**console_kwargs)
    logger.debug(f"Saving replay to {console_kwargs['replay_dir']}")

    # Create our Controller object
    #   The controller is the second primary object your bot will interact with
    #   Your controller is your way of sending button presses to the game, whether
    #   virtual or physical.
    controller = melee.Controller(console=console, port=args.port, type=melee.ControllerType.STANDARD)

    controller_opponent = melee.Controller(console=console, port=args.opponent, type=melee.ControllerType.STANDARD)

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(sig, frame) -> None:
        console.stop()
        if log:
            log.writelog()
            logger.debug("")  # because the ^C will be on the terminal
            logger.debug("Log file created: " + log.filename)
        logger.debug("Shutting down cleanly...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run the console
    console.run(iso_path=ISO_PATH)

    # Connect to the console
    logger.debug("Connecting to console...")
    if not console.connect():
        logger.debug("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    logger.debug("Console connected")

    # Plug our controller in
    #   Due to how named pipes work, this has to come AFTER running dolphin
    #   NOTE: If you're loading a movie file, don't connect the controller,
    #   dolphin will hang waiting for input and never receive it
    logger.debug("Connecting controller 1 to console...")
    if not controller.connect():
        logger.debug("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    logger.debug("Controller 1 connected")
    logger.debug("Connecting controller 2 to console...")
    if not controller_opponent.connect():
        logger.debug("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    logger.debug("Controller 2 connected")

    costume = 0

    menu_helper = MatchupMenuHelper(
        controller_1=controller,
        controller_2=controller_opponent,
        character_1=melee.Character.FOX,
        character_2=melee.Character.FOX,
        stage=melee.Stage.BATTLEFIELD,
        opponent_cpu_level=0,
    )

    # Main loop
    i = 0
    match_started = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor, console_manager(console, log):
        while True:
            # Wrap `console.step()` in a thread with timeout
            future = executor.submit(console.step)
            try:
                gamestate = future.result(timeout=2.0)
            except concurrent.futures.TimeoutError:
                logger.debug("console.step() timed out")
                raise
            if gamestate is None:
                logger.debug("Gamestate is None")
                continue

            # logger.debug(f"{gamestate.menu_state=}")

            # The console object keeps track of how long your bot is taking to process frames
            #   And can warn you if it's taking too long
            if console.processingtime * 1000 > 12:
                logger.debug("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")

            # What menu are we in?
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                if not match_started:
                    logger.debug("Match started")
                    match_started = True
                # Slippi Online matches assign you a random port once you're in game that's different
                #   than the one you're physically plugged into. This helper will autodiscover what
                #   port we actually are.
                discovered_port = args.port
                if args.connect_code != "":
                    discovered_port = melee.gamestate.port_detector(gamestate, melee.Character.FOX, costume)
                if discovered_port > 0:
                    # NOTE: This is where your AI does all of its stuff!
                    # This line will get hit once per frame, so here is where you read
                    #   in the gamestate and decide what buttons to push on the controller
                    melee.techskill.multishine(ai_state=gamestate.players[discovered_port], controller=controller)
                else:
                    # If the discovered port was unsure, reroll our costume for next time
                    costume = random.randint(0, 4)

                # Log this frame's detailed info if we're in game
                if log:
                    log.logframe(gamestate)
                    log.writeframe()

                i += 1
                if i % 60 == 0:
                    logger.debug(f"Frame {i}")
                if i > 1800:
                    break

            else:
                menu_helper.select_character_and_stage(gamestate)

                # If we're not in game, don't log the frame
                if log:
                    log.skipframe()


if __name__ == "__main__":
    main()
