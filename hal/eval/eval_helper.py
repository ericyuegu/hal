from collections import deque
from typing import DefaultDict
from typing import Dict

import melee
import torch
from loguru import logger

from hal.data.constants import IDX_BY_ACTION
from hal.data.constants import IDX_BY_CHARACTER
from hal.data.constants import IDX_BY_STAGE


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
            logger.info(f"Pressed {button}")
            break
    controller.flush()


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
