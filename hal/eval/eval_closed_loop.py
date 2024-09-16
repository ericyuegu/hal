import argparse
import signal
import sys
from collections import defaultdict
from collections import deque
from pathlib import Path
from typing import Any
from typing import DefaultDict
from typing import Dict

import melee
import torch
from data.stats import load_dataset_stats
from loguru import logger
from melee import enums
from melee.menuhelper import MenuHelper
from tensordict import TensorDict

from hal.data.constants import IDX_BY_ACTION
from hal.data.constants import IDX_BY_CHARACTER
from hal.data.constants import IDX_BY_STAGE
from hal.eval.emulator_paths import LOCAL_CISO_PATH
from hal.eval.emulator_paths import LOCAL_DOLPHIN_HOME_PATH
from hal.eval.emulator_paths import LOCAL_GUI_EMULATOR_PATH
from hal.eval.emulator_paths import LOCAL_HEADLESS_EMULATOR_PATH
from hal.eval.emulator_paths import REMOTE_CISO_PATH
from hal.eval.emulator_paths import REMOTE_DOLPHIN_HOME_PATH
from hal.eval.emulator_paths import REMOTE_EMULATOR_PATH
from hal.training.io import load_model_from_artifact_dir
from hal.training.zoo.preprocess.registry import InputPreprocessRegistry
from hal.training.zoo.preprocess.registry import OutputProcessingRegistry

PLAYER_1_PORT = 1
PLAYER_2_PORT = 2


def get_dolphin_home_path(local: bool) -> str:
    if local:
        return LOCAL_DOLPHIN_HOME_PATH
    else:
        return REMOTE_DOLPHIN_HOME_PATH


def get_emulator_path(local: bool, no_gui: bool) -> str:
    if local:
        if no_gui:
            return LOCAL_HEADLESS_EMULATOR_PATH
        else:
            return LOCAL_GUI_EMULATOR_PATH
    else:
        return REMOTE_EMULATOR_PATH


def get_ciso_path(local: bool) -> str:
    if local:
        return LOCAL_CISO_PATH
    else:
        return REMOTE_CISO_PATH


def get_console_kwargs(local: bool, no_gui: bool, debug: bool) -> Dict[str, Any]:
    headless_console_kwargs = (
        {
            "gfx_backend": "Null",
            "disable_audio": True,
            "use_exi_inputs": True,
            "enable_ffw": False,
        }
        if no_gui
        else {}
    )
    emulator_path = get_emulator_path(local=local, no_gui=no_gui)
    dolphin_home_path = get_dolphin_home_path(local=local)
    Path(dolphin_home_path).mkdir(exist_ok=True, parents=True)
    console_kwargs = {
        "path": emulator_path,
        "is_dolphin": True,
        "dolphin_home_path": dolphin_home_path,
        "tmp_home_directory": False,
        "blocking_input": False,  # TODO(eric): investigate why this is stopping menuhelper
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
                cpu_level=9,
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


def extract_and_append_gamestate(gamestate: melee.GameState, frame_data: DefaultDict[str, deque]) -> None:
    """Extracts and appends gamestate data to sliding window."""
    players = sorted(gamestate.players.items())
    if len(players) != 2:
        raise ValueError(f"Expected 2 players, got {len(players)}")

    frame_data["frame"].append(gamestate.frame)
    frame_data["stage"].append(IDX_BY_STAGE[gamestate.stage])

    for i, (port, player_state) in enumerate(players, start=1):
        prefix = f"p{i}"

        # Player state data
        player_data = {
            "port": port,
            "character": IDX_BY_CHARACTER[player_state.character],
            "stock": player_state.stock,
            "facing": int(player_state.facing),
            "invulnerable": int(player_state.invulnerable),
            "position_x": float(player_state.position.x),
            "position_y": float(player_state.position.y),
            "percent": player_state.percent,
            "shield_strength": player_state.shield_strength,
            "jumps_left": player_state.jumps_left,
            "action": IDX_BY_ACTION[player_state.action],
            "action_frame": player_state.action_frame,
            "invulnerability_left": player_state.invulnerability_left,
            "hitlag_left": player_state.hitlag_left,
            "hitstun_left": player_state.hitstun_frames_left,
            "on_ground": int(player_state.on_ground),
            "speed_air_x_self": player_state.speed_air_x_self,
            "speed_y_self": player_state.speed_y_self,
            "speed_x_attack": player_state.speed_x_attack,
            "speed_y_attack": player_state.speed_y_attack,
            "speed_ground_x_self": player_state.speed_ground_x_self,
        }

        # ECB data
        for ecb in ["bottom", "top", "left", "right"]:
            player_data[f"ecb_{ecb}_x"] = getattr(player_state, f"ecb_{ecb}")[0]
            player_data[f"ecb_{ecb}_y"] = getattr(player_state, f"ecb_{ecb}")[1]

        # Append all player state data
        for key, value in player_data.items():
            frame_data[f"{prefix}_{key}"].append(value)

        # Controller data (from current gamestate)
        controller = gamestate.players[port].controller_state

        # Button data
        buttons = ["A", "B", "X", "Y", "Z", "START", "L", "R", "D_UP"]
        for button in buttons:
            frame_data[f"{prefix}_button_{button.lower()}"].append(
                int(controller.button[getattr(melee.Button, f"BUTTON_{button}")])
            )

        # Stick and shoulder data
        frame_data[f"{prefix}_main_stick_x"].append(float(controller.main_stick[0]))
        frame_data[f"{prefix}_main_stick_y"].append(float(controller.main_stick[1]))
        frame_data[f"{prefix}_c_stick_x"].append(float(controller.c_stick[0]))
        frame_data[f"{prefix}_c_stick_y"].append(float(controller.c_stick[1]))
        frame_data[f"{prefix}_l_shoulder"].append(float(controller.l_shoulder))
        frame_data[f"{prefix}_r_shoulder"].append(float(controller.r_shoulder))


def convert_frame_data_to_tensor_dict(frame_data: DefaultDict[str, deque]) -> TensorDict:
    return TensorDict({k: torch.tensor(v) for k, v in frame_data.items()})


def send_controller_inputs(controller: melee.Controller, inputs: Dict[str, torch.Tensor], idx: int = -1) -> None:
    """
    Press buttons and tilt analog sticks given a dictionary of array-like values (length T for T future time steps).

    Args:
        controller_inputs (Dict[str, torch.Tensor]): Dictionary of array-like values.
        controller (melee.Controller): Controller object.
        idx (int): Index in the arrays to send.
    """
    if idx >= 0:
        assert idx < len(inputs["main_stick_x"])

    controller.tilt_analog(
        melee.Button.BUTTON_MAIN,
        inputs["main_stick_x"][idx].item(),
        inputs["main_stick_y"][idx].item(),
    )
    controller.tilt_analog(
        melee.Button.BUTTON_C,
        inputs["c_stick_x"][idx].item(),
        inputs["c_stick_y"][idx].item(),
    )
    for button, state in inputs.items():
        if button.startswith("button") and button != "button_none" and state[idx].item() == 1:
            controller.press_button(getattr(melee.Button, button.upper()))
            break


def run_episode(local: bool, no_gui: bool, debug: bool, model_dir: str) -> None:
    console_kwargs = get_console_kwargs(local=local, no_gui=no_gui, debug=debug)
    console = melee.Console(**console_kwargs)
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
    console.run(iso_path=get_ciso_path(local), dolphin_user_path=get_dolphin_home_path(local))
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

    model, train_config = load_model_from_artifact_dir(Path(model_dir))
    model.eval()
    preprocess_inputs = InputPreprocessRegistry.get(train_config.embedding.input_preprocessing_fn)
    stats_by_feature_name = load_dataset_stats(train_config.data.stats_path)
    postprocess_outputs = OutputProcessingRegistry.get(train_config.embedding.target_preprocessing_fn)

    # Container for sliding window of model inputs
    frame_data: DefaultDict[str, deque] = defaultdict(lambda: deque(maxlen=train_config.data.input_len))

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

        print(f"frame {i}: {gamestate.menu_state=} {gamestate.submenu=}")
        active_buttons = tuple(button for button, state in controller_1.current.button.items() if state == True)
        print(f"Controller 1: {active_buttons=}")
        active_buttons = tuple(button for button, state in controller_2.current.button.items() if state == True)
        print(f"Controller 2: {active_buttons=}")

        # What menu are we in?
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            extract_and_append_gamestate(gamestate=gamestate, frame_data=frame_data)
            frame_data_td = convert_frame_data_to_tensor_dict(frame_data)
            model_inputs = preprocess_inputs(frame_data_td, train_config.data, "p1", stats_by_feature_name)
            outputs: TensorDict = model(model_inputs)
            controller_inputs = postprocess_outputs(outputs)
            send_controller_inputs(controller_1, controller_inputs)

            melee.techskill.multishine(ai_state=gamestate.players[PLAYER_2_PORT], controller=controller_2)
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
                stage_selected=melee.Stage.BATTLEFIELD,
            )

            # If we're not in game, don't log the frame
            if log:
                log.skipframe()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Melee in emulator")
    parser.add_argument("--local", action="store_true", help="Run in local mode")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--debug", action="store_true", help="Run with debug mode")
    parser.add_argument("--model_dir", type=str, help="Path to model directory")
    args = parser.parse_args()
    run_episode(local=args.local, no_gui=args.no_gui, debug=args.debug, model_dir=args.model_dir)
