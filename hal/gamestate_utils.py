from typing import Any
from typing import DefaultDict
from typing import MutableSequence
from typing import Optional

import melee

from hal.data.constants import IDX_BY_ACTION
from hal.data.constants import IDX_BY_CHARACTER
from hal.data.constants import IDX_BY_STAGE

FrameData = DefaultDict[str, MutableSequence[Any]]


def extract_and_append_gamestate_inplace(
    frame_data_by_field: FrameData,
    curr_gamestate: melee.GameState,
    next_gamestate: Optional[melee.GameState] = None,
    replay_uuid: Optional[int] = None,
) -> FrameData:
    """
    Extract gamestate and controller inputs and store in-place in `frame_data`.

    Groups values for each field across frames.

    Skips controller data if `next_gamestate` is None.

    Controller state is stored in GameState objects with resultant gamestate
    from sending that controller input. We need to read next gamestate to
    correctly pair controller inputs with current gamestate for sequential
    modeling, i.e. what buttons to press next *given the current frame*.
    """
    players = sorted(curr_gamestate.players.items())
    assert len(players) == 2, f"Expected 2 players, got {len(players)}"

    if replay_uuid is not None:
        # Duplicate replay_uuid across frames for preprocessing simplicity
        frame_data_by_field["replay_uuid"].append(replay_uuid)

    frame_data_by_field["frame"].append(curr_gamestate.frame)
    frame_data_by_field["stage"].append(IDX_BY_STAGE[curr_gamestate.stage])

    for i, (port, player_state) in enumerate(players, start=1):
        player_name = f"p{i}"

        # Player / gamestate data
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

        for ecb in ["bottom", "top", "left", "right"]:
            player_data[f"ecb_{ecb}_x"] = getattr(player_state, f"ecb_{ecb}")[0]
            player_data[f"ecb_{ecb}_y"] = getattr(player_state, f"ecb_{ecb}")[1]

        for player_state_field, value in player_data.items():
            frame_data_by_field[f"{player_name}_{player_state_field}"].append(value)

        # Skip controller data if next gamestate is None
        if next_gamestate is None:
            continue

        # Controller data (from next gamestate)
        controller = next_gamestate.players[port].controller_state
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
