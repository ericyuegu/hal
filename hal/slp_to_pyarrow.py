import multiprocessing as mp
import os
from typing import List

import attr
import melee
import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger

schema = pa.schema(
    [
        ("replay_uuid", pa.int64()),
        ("frame", pa.int32()),
        ("stage", pa.int8()),
        ("p1_port", pa.int8()),
        ("p1_character", pa.int8()),
        ("p1_position_x", pa.float32()),
        ("p1_position_y", pa.float32()),
        ("p1_percent", pa.float32()),
        ("p1_shield_strength", pa.float32()),
        ("p1_stock", pa.int8()),
        ("p1_facing", pa.bool_()),
        ("p1_action", pa.int16()),
        ("p1_action_frame", pa.int16()),
        ("p1_invulnerable", pa.bool_()),
        ("p1_invulnerability_left", pa.int16()),
        ("p1_hitlag_left", pa.int16()),
        ("p1_hitstun_left", pa.int16()),
        ("p1_jumps_left", pa.int8()),
        ("p1_on_ground", pa.bool_()),
        ("p1_speed_air_x_self", pa.float32()),
        ("p1_speed_y_self", pa.float32()),
        ("p1_speed_x_attack", pa.float32()),
        ("p1_speed_y_attack", pa.float32()),
        ("p1_speed_ground_x_self", pa.float32()),
        ("p2_port", pa.int8()),
        ("p2_character", pa.int8()),
        ("p2_position_x", pa.float32()),
        ("p2_position_y", pa.float32()),
        ("p2_percent", pa.float32()),
        ("p2_shield_strength", pa.float32()),
        ("p2_stock", pa.int8()),
        ("p2_facing", pa.bool_()),
        ("p2_action", pa.int16()),
        ("p2_action_frame", pa.int16()),
        ("p2_invulnerable", pa.bool_()),
        ("p2_invulnerability_left", pa.int16()),
        ("p2_hitlag_left", pa.int16()),
        ("p2_hitstun_left", pa.int16()),
        ("p2_jumps_left", pa.int8()),
        ("p2_on_ground", pa.bool_()),
        ("p2_speed_air_x_self", pa.float32()),
        ("p2_speed_y_self", pa.float32()),
        ("p2_speed_x_attack", pa.float32()),
        ("p2_speed_y_attack", pa.float32()),
        ("p2_speed_ground_x_self", pa.float32()),
    ]
)

# Simplified enum mappings
EXCLUDED_STAGES = (
    "NO_STAGE",
    "RANDOM_STAGE",
)
IDX_BY_STAGE = {
    stage: i for i, stage in enumerate(stage for stage in melee.Stage if stage.name not in EXCLUDED_STAGES)
}
STAGE_BY_IDX = {i: stage.name for stage, i in IDX_BY_STAGE.items()}

EXCLUDED_CHARACTERS = ("NANA", "WIREFRAME_MALE", "WIREFRAME_FEMALE", "GIGA_BOWSER", "SANDBAG", "UNKNOWN_CHARACTER")
IDX_BY_CHARACTER = {
    char: i for i, char in enumerate(char for char in melee.Character if char.name not in EXCLUDED_CHARACTERS)
}
CHARACTER_BY_IDX = {i: char.name for char, i in IDX_BY_CHARACTER.items()}

IDX_BY_ACTION = {action: i for i, action in enumerate(melee.Action)}
ACTION_BY_IDX = {i: action.name for action, i in IDX_BY_ACTION.items()}


@attr.s(auto_attribs=True, frozen=True)
class FrameData:
    replay_uuid: int
    frame: int
    stage: int

    p1_port: int
    p1_character: int
    p1_position_x: float
    p1_position_y: float
    p1_percent: float
    p1_shield_strength: float
    p1_stock: int
    p1_facing: bool
    p1_action: int
    p1_action_frame: int
    p1_action_frame: int
    p1_invulnerable: bool
    p1_invulnerability_left: int
    p1_hitlag_left: int
    p1_hitstun_left: int
    p1_jumps_left: int
    p1_on_ground: bool
    p1_speed_air_x_self: float
    p1_speed_y_self: float
    p1_speed_x_attack: float
    p1_speed_y_attack: float
    p1_speed_ground_x_self: float

    p2_port: int
    p2_character: int
    p2_position_x: float
    p2_position_y: float
    p2_percent: float
    p2_shield_strength: float
    p2_stock: int
    p2_facing: bool
    p2_action: int
    p2_action_frame: int
    p2_action_frame: int
    p2_invulnerable: bool
    p2_invulnerability_left: int
    p2_hitlag_left: int
    p2_hitstun_left: int
    p2_jumps_left: int
    p2_on_ground: bool
    p2_speed_air_x_self: float
    p2_speed_y_self: float
    p2_speed_x_attack: float
    p2_speed_y_attack: float
    p2_speed_ground_x_self: float


def extract_frame_data(gamestate: melee.GameState, replay_uuid: int) -> FrameData:
    players = sorted(gamestate.players.items())
    if len(players) != 2:
        raise ValueError(f"Expected 2 players, got {len(players)}")

    p1_port, p1 = players[0]
    p2_port, p2 = players[1]

    return FrameData(
        replay_uuid=replay_uuid,
        frame=gamestate.frame,
        stage=gamestate.stage.value,
        p1_port=p1_port,
        p1_character=p1.character.value,
        p1_stock=p1.stock,
        p1_facing=p1.facing,
        p1_invulnerable=p1.invulnerable,
        p1_position_x=float(p1.position.x),
        p1_position_y=float(p1.position.y),
        p1_percent=p1.percent,
        p1_shield_strength=p1.shield_strength,
        p1_jumps_left=p1.jumps_left,
        p1_action=p1.action.value,
        p1_action_frame=p1.action_frame,
        p1_invulnerability_left=p1.invulnerability_left,
        p1_hitlag_left=p1.hitlag_left,
        p1_hitstun_left=p1.hitstun_frames_left,
        p1_on_ground=p1.on_ground,
        p1_speed_air_x_self=p1.speed_air_x_self,
        p1_speed_y_self=p1.speed_y_self,
        p1_speed_x_attack=p1.speed_x_attack,
        p1_speed_y_attack=p1.speed_y_attack,
        p1_speed_ground_x_self=p1.speed_ground_x_self,
        p2_port=p2_port,
        p2_character=p2.character.value,
        p2_stock=p2.stock,
        p2_facing=p2.facing,
        p2_invulnerable=p2.invulnerable,
        p2_position_x=float(p2.position.x),
        p2_position_y=float(p2.position.y),
        p2_percent=p2.percent,
        p2_shield_strength=p2.shield_strength,
        p2_jumps_left=p2.jumps_left,
        p2_action=p2.action.value,
        p2_action_frame=p2.action_frame,
        p2_invulnerability_left=p2.invulnerability_left,
        p2_hitlag_left=p2.hitlag_left,
        p2_hitstun_left=p2.hitstun_frames_left,
        p2_on_ground=p2.on_ground,
        p2_speed_air_x_self=p2.speed_air_x_self,
        p2_speed_y_self=p2.speed_y_self,
        p2_speed_x_attack=p2.speed_x_attack,
        p2_speed_y_attack=p2.speed_y_attack,
        p2_speed_ground_x_self=p2.speed_ground_x_self,
    )


def process_replay(replay_path: str) -> List[FrameData]:
    console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
    try:
        console.connect()
    except Exception as e:
        logger.error(f"Error connecting to console: {e}")
        return []

    frame_data = []
    replay_uuid = hash(replay_path)

    while True:
        try:
            gamestate = console.step()
            if gamestate is None:
                break
            frame_data.append(extract_frame_data(gamestate, replay_uuid))
        except Exception as e:
            logger.debug(f"Could not read gamestate from {replay_path}: {e}")
            break

    return frame_data


def write_dataset(data: List[FrameData], output_dir: str) -> None:
    table = pa.Table.from_pylist([attr.asdict(frame) for frame in data], schema=schema)
    ds.write_dataset(table, output_dir, format="parquet", partitioning=["stage"])


def process_and_write(replay_path: str, output_dir: str) -> None:
    data = process_replay(replay_path)
    if data:
        write_dataset(data, os.path.join(output_dir, os.path.basename(replay_path)))
    else:
        logger.warning(f"No data extracted from {replay_path}")


def process_replays(replay_paths: List[str], output_dir: str) -> None:
    with mp.Pool() as pool:
        pool.starmap(process_and_write, [(path, output_dir) for path in replay_paths])


if __name__ == "__main__":
    # Add command-line argument parsing here if needed
    # process_replays(replay_paths, output_path)
    pass
