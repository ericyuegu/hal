import signal
import sys
from pathlib import Path
from typing import Final

import melee
from melee import enums
from melee.menuhelper import MenuHelper

from hal.emulator_paths import LOCAL_CISO_PATH
from hal.emulator_paths import LOCAL_GUI_EMULATOR_PATH

DOLPHIN_HOME_PATH: Final[Path] = Path("/home/egu/Slippi")
PLAYER_1_PORT = 1
PLAYER_2_PORT = 4


def self_play_menu_helper(
    gamestate: melee.GameState,
    controller_1: melee.Controller,
    controller_2: melee.Controller,
    character_1: melee.Character,
    character_2: melee.Character,
    stage_selected: melee.Stage,
) -> None:
    if gamestate.menu_state == enums.Menu.MAIN_MENU:
        MenuHelper.choose_versus_mode(gamestate=gamestate, controller=controller_1)
    # If we're at the character select screen, choose our character
    elif gamestate.menu_state == enums.Menu.CHARACTER_SELECT:
        player_1 = gamestate.players[controller_1.port]
        player_1_character_selected = player_1.character == character_1
        player_2 = gamestate.players[controller_2.port]
        player_2_character_selected = player_2.character == character_2

        print(f"{player_1_character_selected=}")
        print(f"{player_2_character_selected=}")
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
                cpu_level=0,
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


def run_episode() -> None:
    DOLPHIN_HOME_PATH.mkdir(exist_ok=True)
    console = melee.Console(
        path=LOCAL_GUI_EMULATOR_PATH,
        is_dolphin=True,
        dolphin_home_path=str(DOLPHIN_HOME_PATH),
        tmp_home_directory=False,
        blocking_input=True,
        use_exi_inputs=False,
        enable_ffw=False,
    )

    log = melee.Logger()

    # Create our Controller object
    #   The controller is the second primary object your bot will interact with
    #   Your controller is your way of sending button presses to the game, whether
    #   virtual or physical.
    controller_1 = melee.Controller(console=console, port=PLAYER_1_PORT, type=melee.ControllerType.STANDARD)
    controller_2 = melee.Controller(console=console, port=PLAYER_2_PORT, type=melee.ControllerType.STANDARD)

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(sig, frame) -> None:
        console.stop()
        log.writelog()
        print("")  # because the ^C will be on the terminal
        print("Log file created: " + log.filename)
        print("Shutting down cleanly...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run the console
    console.run(iso_path=LOCAL_CISO_PATH, dolphin_user_path=str(DOLPHIN_HOME_PATH))

    # Connect to the console
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("Console connected")

    # Plug our controller in
    #   Due to how named pipes work, this has to come AFTER running dolphin
    #   NOTE: If you're loading a movie file, don't connect the controller,
    #   dolphin will hang waiting for input and never receive it
    print("Connecting controller 1 to console...")
    if not controller_1.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller 1 connected")
    print("Connecting controller 2 to console...")
    if not controller_2.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller 2 connected")

    costume = 0
    framedata = melee.framedata.FrameData()

    # Main loop
    i = 0
    while i < 10000:
        # "step" to the next frame
        gamestate = console.step()
        if gamestate is None:
            continue

        # The console object keeps track of how long your bot is taking to process frames
        #   And can warn you if it's taking too long
        if console.processingtime * 1000 > 12:
            print("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")

        print(f"frame {i}: {gamestate.menu_state=}")
        active_buttons = tuple(button for button, state in controller_1.current.button.items() if state == True)
        print(f"Controller 1: {active_buttons=}")
        active_buttons = tuple(button for button, state in controller_2.current.button.items() if state == True)
        print(f"Controller 2: {active_buttons=}")

        # What menu are we in?
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            melee.techskill.multishine(ai_state=gamestate.players[PLAYER_1_PORT], controller=controller_1)
            melee.techskill.multishine(ai_state=gamestate.players[PLAYER_2_PORT], controller=controller_2)
            print(f"Frame {i}")
            i += 1

            # Log this frame's detailed info if we're in game
            if log:
                log.logframe(gamestate)
                log.writeframe()

        else:
            self_play_menu_helper(
                gamestate=gamestate,
                controller_1=controller_1,
                controller_2=controller_2,
                character_1=melee.Character.FOX,
                character_2=melee.Character.FOX,
                stage_selected=melee.Stage.YOSHIS_STORY,
            )

            # If we're not in game, don't log the frame
            if log:
                log.skipframe()


if __name__ == "__main__":
    run_episode()
