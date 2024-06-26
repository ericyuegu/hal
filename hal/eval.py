from pathlib import Path
import signal
import sys

from datetime import datetime
from typing import Final
import melee

from hal.emulator_paths import REMOTE_CISO_PATH
from hal.emulator_paths import REMOTE_EMULATOR_PATH

REPLAY_OUTPUT_PATH: Final[Path] = Path("/opt/projects/hal/replays")


def run_episode() -> None:
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    replay_output_path = REPLAY_OUTPUT_PATH / timestamp
    console = melee.Console(
        path=REMOTE_EMULATOR_PATH,
        is_dolphin=True,
        dolphin_home_path=str(replay_output_path),
        tmp_home_directory=False,
        gfx_backend="Null",
        disable_audio=True,
        use_exi_inputs=True,
        enable_ffw=True,
    )

    log = melee.Logger()

    # Create our Controller object
    #   The controller is the second primary object your bot will interact with
    #   Your controller is your way of sending button presses to the game, whether
    #   virtual or physical.
    PLAYER_1_PORT = 1
    PLAYER_2_PORT = 2
    controller = melee.Controller(console=console, port=PLAYER_1_PORT, type=melee.ControllerType.STANDARD)
    controller_opponent = melee.Controller(console=console, port=PLAYER_2_PORT, type=melee.ControllerType.GCN_ADAPTER)

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
    console.run(iso_path=REMOTE_CISO_PATH, dolphin_user_path=)

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
    print("Connecting controller to console...")
    if not controller.connect():
        print("ERROR: Failed to connect the controller.")
        sys.exit(-1)
    print("Controller connected")

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

        # What menu are we in?
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            print(f"Frame {i}")
            melee.techskill.multishine(ai_state=gamestate.players[PLAYER_1_PORT], controller=controller)

            # Log this frame's detailed info if we're in game
            if log:
                log.logframe(gamestate)
                log.writeframe()

            i += 1
        else:
            melee.MenuHelper.menu_helper_simple(
                gamestate,
                controller,
                character_selected=melee.Character.FOX,
                stage_selected=melee.Stage.YOSHIS_STORY,
                connect_code="",
                costume=costume,
                autostart=True,
                swag=False,
            )

            # If we're not in game, don't log the frame
            if log:
                log.skipframe()


if __name__ == "__main__":
    run_episode()
