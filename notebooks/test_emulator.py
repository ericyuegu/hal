# %%
from collections import defaultdict
from pathlib import Path

import melee
import melee.enums as enums
import torch
from loguru import logger
from tensordict import TensorDict

from hal.emulator_helper import EmulatorManager
from hal.emulator_helper import find_open_udp_ports
from hal.gamestate_utils import extract_and_append_gamestate_inplace


def multishine(ai_state: melee.PlayerState) -> TensorDict:
    """
    Press buttons and tilt analog sticks given a dictionary of array-like values (length T for T future time steps).

    Args:
        controller_inputs (Dict[str, torch.Tensor]): Dictionary of array-like values.
        controller (melee.Controller): Controller object.
        idx (int): Index in the arrays to send.
    """
    inputs = {
        "main_stick_x": [0.5],
        "main_stick_y": [0.5],
        "c_stick_x": [0.5],
        "c_stick_y": [0.5],
        "button": [5],
    }

    def convert_to_tensordict(list_dict: dict) -> TensorDict:
        for k, v in list_dict.items():
            list_dict[k] = torch.tensor(v)
        return TensorDict(list_dict, batch_size=(1,))

    if ai_state.action == enums.Action.STANDING:
        inputs["button"] = [1]
        inputs["main_stick_y"] = [0]
        return convert_to_tensordict(inputs)

    if ai_state.action == enums.Action.KNEE_BEND:
        if ai_state.action_frame == 3:
            inputs["button"] = [1]
            inputs["main_stick_y"] = [0]
            return convert_to_tensordict(inputs)

        return convert_to_tensordict(inputs)

    shine_start = ai_state.action == enums.Action.DOWN_B_STUN or ai_state.action == enums.Action.DOWN_B_GROUND_START

    if shine_start and ai_state.action_frame >= 4 and ai_state.on_ground:
        # Jump out of shine
        inputs["button"] = [2]
        return convert_to_tensordict(inputs)

    if ai_state.action == enums.Action.DOWN_B_GROUND:
        inputs["button"] = [2]
        return convert_to_tensordict(inputs)

    return convert_to_tensordict(inputs)


# %%
port = find_open_udp_ports(1)
emulator_manager = EmulatorManager(
    udp_port=port[0],
    player="p1",
    replay_dir=Path("/tmp/slippi_replays"),
    max_steps=30000,
    enable_ffw=True,
    debug=True,
    opponent_cpu_level=9,
)
ego_controller = emulator_manager.ego_controller
gamestate_generator = emulator_manager.run_game()
gamestate = next(gamestate_generator)
i = 0
while gamestate is not None:
    # td = extract_gamestate_as_tensordict(gamestate)
    controller_inputs = multishine(ai_state=gamestate.players[1])
    # logger.debug(f"Player action {gamestate.players[1].action}")
    if i % 60 == 0:
        logger.debug(f"Outer loop {i}")
    # if controller_inputs is None:
    #     logger.error("multishine returned None")
    try:
        gamestate = gamestate_generator.send(controller_inputs)
    except StopIteration:
        break
    i += 1

# %%
replay_path = "/tmp/slippi_replays/Game_20250203T162044.slp"
replay_uuid = hash(replay_path)
frame_data = defaultdict(list)
console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
console.connect()
next_gamestate = console.step()
try:
    while next_gamestate is not None:
        curr_gamestate = next_gamestate
        next_gamestate = console.step()
        if next_gamestate is None:
            break

        frame_data = extract_and_append_gamestate_inplace(
            frame_data_by_field=frame_data,
            curr_gamestate=curr_gamestate,
            next_gamestate=next_gamestate,
            replay_uuid=replay_uuid,
        )
except Exception as e:
    print(f"Error processing replay {replay_path}: {e}")
finally:
    console.stop()


# %%
frame_data["p1_action"]

# # %%
# def multishine(ai_state: melee.PlayerState, controller: melee.Controller) -> None:
#     # If standing, shine
#     if ai_state.action == enums.Action.STANDING:
#         controller.press_button(enums.Button.BUTTON_B)
#         controller.tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0)
#         return

#     # Shine on frame 3 of knee bend, else nothing
#     if ai_state.action == enums.Action.KNEE_BEND:
#         if ai_state.action_frame == 3:
#             controller.press_button(enums.Button.BUTTON_B)
#             controller.tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0)
#             return
#         controller.release_all()
#         return

#     shine_start = ai_state.action == enums.Action.DOWN_B_STUN or ai_state.action == enums.Action.DOWN_B_GROUND_START

#     # Jump out of shine
#     if shine_start and ai_state.action_frame >= 4 and ai_state.on_ground:
#         controller.press_button(enums.Button.BUTTON_Y)
#         return

#     if ai_state.action == enums.Action.DOWN_B_GROUND:
#         controller.press_button(enums.Button.BUTTON_Y)
#         return

#     controller.release_all()
