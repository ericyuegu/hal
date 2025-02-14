from collections import defaultdict
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import MutableSequence
from typing import Optional

import melee
import torch
from tensordict import TensorDict

from hal.constants import IDX_BY_ACTION
from hal.constants import IDX_BY_CHARACTER
from hal.constants import IDX_BY_STAGE
from hal.constants import INCLUDED_CHARACTERS
from hal.constants import INCLUDED_STAGES

FrameData = DefaultDict[str, MutableSequence[Any]]


def extract_gamestate_as_tensordict(gamestate: melee.GameState) -> TensorDict:
    frame_data: FrameData = defaultdict(list)
    extract_and_append_gamestate_inplace(frame_data, gamestate)
    return TensorDict({k: torch.tensor(v) for k, v in frame_data.items()}, batch_size=(1,))


def extract_player_state(player_state: melee.PlayerState) -> Dict[str, Any]:
    player_data = {
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

    for ecb in ["bottom", "top", "left", "right"]:
        player_data[f"ecb_{ecb}_x"] = getattr(player_state, f"ecb_{ecb}")[0]
        player_data[f"ecb_{ecb}_y"] = getattr(player_state, f"ecb_{ecb}")[1]
    return player_data


def extract_and_append_gamestate_inplace(
    frame_data_by_field: FrameData,
    curr_gamestate: melee.GameState,
    next_gamestate: Optional[melee.GameState] = None,
    replay_uuid: Optional[int] = None,
) -> FrameData:
    """
    Extract gamestate and controller inputs and store in-place in `frame_data`.

    Groups values for each field across frames.

    **Test time behavior**
    If `next_gamestate` is None, extracts controller data from `curr_gamestate`

    Controller state is stored in `melee.GameState` objects, where gamestate is
    resultant from sending that controller input. We need to read next
    gamestate in order to correctly pair controller inputs with curr gamestate
    for sequential modeling, i.e. what buttons to press next *given the current
    frame*.
    """
    players = sorted(curr_gamestate.players.items())
    assert len(players) == 2, f"Expected 2 players, got {len(players)}"
    assert curr_gamestate.stage.name in INCLUDED_STAGES, f"Stage {curr_gamestate.stage} not valid"

    if replay_uuid is not None:
        # Duplicate replay_uuid across frames for preprocessing simplicity
        frame_data_by_field["replay_uuid"].append(replay_uuid)

    frame_data_by_field["frame"].append(curr_gamestate.frame)
    frame_data_by_field["stage"].append(IDX_BY_STAGE[curr_gamestate.stage])

    for i, (port, player_state) in enumerate(players, start=1):
        player_relative = f"p{i}"
        assert player_state.character.name in INCLUDED_CHARACTERS, f"Character {player_state.character} not valid"

        # Player / gamestate data
        player_data = extract_player_state(player_state)

        # Handle Ice Climbers' Nana data (empirically appears in about 5% of games)
        if player_state.nana is not None:
            nana_data = extract_player_state(player_state.nana)
        else:
            # WARNING: duplicates all keys for player state, place unique items after
            nana_data = {k: None for k in player_data.keys()}
        player_data.update({f"nana_{k}": v for k, v in nana_data.items()})

        player_data["port"] = port
        for feature_name, value in player_data.items():
            frame_data_by_field[f"{player_relative}_{feature_name}"].append(value)

    if next_gamestate is None:
        extract_controller_inputs_inplace(frame_data_by_field=frame_data_by_field, gamestate=curr_gamestate)
    else:
        extract_controller_inputs_inplace(frame_data_by_field=frame_data_by_field, gamestate=next_gamestate)

    return frame_data_by_field


def extract_controller_inputs_inplace(
    frame_data_by_field: FrameData,
    gamestate: melee.GameState,
) -> FrameData:
    """Extract controller inputs from gamestate and store in-place in `frame_data`."""
    players = sorted(gamestate.players.items())
    assert len(players) == 2, f"Expected 2 players, got {len(players)}"

    for i, (_, player_state) in enumerate(players, start=1):
        player_name = f"p{i}"

        controller = player_state.controller_state
        buttons = ["A", "B", "X", "Y", "Z", "START", "L", "R", "D_UP"]
        for button in buttons:
            frame_data_by_field[f"{player_name}_button_{button.lower()}"].append(
                int(controller.button[getattr(melee.Button, f"BUTTON_{button}")])
            )
        frame_data_by_field[f"{player_name}_main_stick_x"].append(float(controller.main_stick[0]))
        frame_data_by_field[f"{player_name}_main_stick_y"].append(float(controller.main_stick[1]))
        frame_data_by_field[f"{player_name}_c_stick_x"].append(float(controller.c_stick[0]))
        frame_data_by_field[f"{player_name}_c_stick_y"].append(float(controller.c_stick[1]))
        frame_data_by_field[f"{player_name}_l_shoulder"].append(float(controller.l_shoulder))
        frame_data_by_field[f"{player_name}_r_shoulder"].append(float(controller.r_shoulder))

    return frame_data_by_field
