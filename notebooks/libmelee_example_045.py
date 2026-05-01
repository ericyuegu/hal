#!/usr/bin/python3
import argparse
import signal
import sys

import melee

# This example program demonstrates how to use the Melee API to run a console,
#   setup controllers, and send button presses over to a console


def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError(
            "%s is an invalid controller port. \
                                         Must be 1, 2, 3, or 4."
            % value
        )
    return ivalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of libmelee in action")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode. Creates a CSV of all game states")
    parser.add_argument("--address", "-a", default="127.0.0.1", help="IP address of Slippi/Wii")
    parser.add_argument("--dolphin_executable_path", "-e", default=None, help="The directory where dolphin is")
    parser.add_argument("--iso", default=None, type=str, help="Path to melee iso.")

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
    console = melee.Console(
        path=args.dolphin_executable_path,
        slippi_address=args.address,
        logger=log,
        save_replays=args.debug,
        fullscreen=False,
    )

    # Create our Controller object
    #   The controller is the second primary object your bot will interact with
    #   Your controller is your way of sending button presses to the game, whether
    #   virtual or physical.

    ports = [1, 2]

    controllers = {
        port: melee.Controller(console=console, port=port, type=melee.ControllerType.STANDARD) for port in ports
    }

    # This isn't necessary, but makes it so that Dolphin will get killed when you ^C
    def signal_handler(sig, frame):
        for controller in controllers.values():
            controller.disconnect()
        console.stop()
        if args.debug:
            log.writelog()
            print("")  # because the ^C will be on the terminal
            print("Log file created: " + log.filename)
        print("Shutting down cleanly...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run the console
    console.run(iso_path=args.iso)

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
    for controller in controllers.values():
        if not controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
    print("Controller connected")

    costume = 0
    framedata = melee.framedata.FrameData()

    menu_helper = melee.MenuHelper()

    # Main loop
    for _ in range(10 * 60):  # Run for 10 seconds
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
            for port, controller in controllers.items():
                # NOTE: This is where your AI does all of its stuff!
                # This line will get hit once per frame, so here is where you read
                #   in the gamestate and decide what buttons to push on the controller
                melee.techskill.multishine(ai_state=gamestate.players[port], controller=controller)

            # Log this frame's detailed info if we're in game
            if log:
                log.logframe(gamestate)
                log.writeframe()
        else:
            for port, controller in controllers.items():
                menu_helper.menu_helper_simple(
                    gamestate,
                    controller,
                    melee.Character.FOX,
                    melee.Stage.YOSHIS_STORY,
                    costume=port,
                    autostart=port == 1,
                    swag=False,
                )

            # If we're not in game, don't log the frame
            if log:
                log.skipframe()

    console.stop()
