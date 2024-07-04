import argparse
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import attr
import melee
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from hal.data.constants import IDX_BY_ACTION
from hal.data.constants import IDX_BY_CHARACTER
from hal.data.constants import IDX_BY_STAGE
from hal.data.primitives import SCHEMA

ControllerData = Dict[str, Any]
FrameData = Dict[str, Any]


def extract_single_frame(gamestate: melee.GameState, replay_uuid: int) -> Tuple[FrameData, ControllerData]:
    """Extracts gamestate and controller data from 1 frame of replay."""
    players = sorted(gamestate.players.items())
    if len(players) != 2:
        raise ValueError(f"Expected 2 players, got {len(players)}")

    p1_port, p1 = players[0]
    p2_port, p2 = players[1]

    # Skip pre-game frames
    if gamestate.frame < 0:
        return {}, {}

    gamestate_frame = dict(
        # Metadata
        replay_uuid=replay_uuid,
        frame=gamestate.frame,
        # Stage
        stage=IDX_BY_STAGE[gamestate.stage],
        # Player 1 state
        p1_port=p1_port,
        p1_character=IDX_BY_CHARACTER[p1.character],
        p1_stock=p1.stock,
        p1_facing=int(p1.facing),
        p1_invulnerable=int(p1.invulnerable),
        p1_position_x=float(p1.position.x),
        p1_position_y=float(p1.position.y),
        p1_percent=p1.percent,
        p1_shield_strength=p1.shield_strength,
        p1_jumps_left=p1.jumps_left,
        p1_action=IDX_BY_ACTION[p1.action],
        p1_action_frame=p1.action_frame,
        p1_invulnerability_left=p1.invulnerability_left,
        p1_hitlag_left=p1.hitlag_left,
        p1_hitstun_left=p1.hitstun_frames_left,
        p1_on_ground=int(p1.on_ground),
        p1_speed_air_x_self=p1.speed_air_x_self,
        p1_speed_y_self=p1.speed_y_self,
        p1_speed_x_attack=p1.speed_x_attack,
        p1_speed_y_attack=p1.speed_y_attack,
        p1_speed_ground_x_self=p1.speed_ground_x_self,
        p1_ecb_bottom_x=p1.ecb_bottom[0],
        p1_ecb_bottom_y=p1.ecb_bottom[1],
        p1_ecb_top_x=p1.ecb_top[0],
        p1_ecb_top_y=p1.ecb_top[1],
        p1_ecb_left_x=p1.ecb_left[0],
        p1_ecb_left_y=p1.ecb_left[1],
        p1_ecb_right_x=p1.ecb_right[0],
        p1_ecb_right_y=p1.ecb_right[1],
        # Player 2 state
        p2_port=p2_port,
        p2_character=IDX_BY_CHARACTER[p2.character],
        p2_stock=p2.stock,
        p2_facing=int(p2.facing),
        p2_invulnerable=int(p2.invulnerable),
        p2_position_x=float(p2.position.x),
        p2_position_y=float(p2.position.y),
        p2_percent=p2.percent,
        p2_shield_strength=p2.shield_strength,
        p2_jumps_left=p2.jumps_left,
        p2_action=IDX_BY_ACTION[p2.action],
        p2_action_frame=p2.action_frame,
        p2_invulnerability_left=p2.invulnerability_left,
        p2_hitlag_left=p2.hitlag_left,
        p2_hitstun_left=p2.hitstun_frames_left,
        p2_on_ground=int(p2.on_ground),
        p2_speed_air_x_self=p2.speed_air_x_self,
        p2_speed_y_self=p2.speed_y_self,
        p2_speed_x_attack=p2.speed_x_attack,
        p2_speed_y_attack=p2.speed_y_attack,
        p2_speed_ground_x_self=p2.speed_ground_x_self,
        p2_ecb_bottom_x=p2.ecb_bottom[0],
        p2_ecb_bottom_y=p2.ecb_bottom[1],
        p2_ecb_top_x=p2.ecb_top[0],
        p2_ecb_top_y=p2.ecb_top[1],
        p2_ecb_left_x=p2.ecb_left[0],
        p2_ecb_left_y=p2.ecb_left[1],
        p2_ecb_right_x=p2.ecb_right[0],
        p2_ecb_right_y=p2.ecb_right[1],
    )

    p1_controller = p1.controller_state
    p2_controller = p2.controller_state

    controller_frame = dict(
        # Player 1
        p1_button_a=int(p1_controller.button[melee.Button.BUTTON_A]),
        p1_button_b=int(p1_controller.button[melee.Button.BUTTON_B]),
        p1_button_x=int(p1_controller.button[melee.Button.BUTTON_X]),
        p1_button_y=int(p1_controller.button[melee.Button.BUTTON_Y]),
        p1_button_z=int(p1_controller.button[melee.Button.BUTTON_Z]),
        p1_button_start=int(p1_controller.button[melee.Button.BUTTON_START]),
        p1_button_d_up=int(p1_controller.button[melee.Button.BUTTON_D_UP]),
        p1_button_l=int(p1_controller.button[melee.Button.BUTTON_L]),
        p1_button_r=int(p1_controller.button[melee.Button.BUTTON_R]),
        p1_main_stick_x=float(p1_controller.main_stick[0]),
        p1_main_stick_y=float(p1_controller.main_stick[1]),
        p1_c_stick_x=float(p1_controller.c_stick[0]),
        p1_c_stick_y=float(p1_controller.c_stick[1]),
        p1_l_shoulder=float(p1_controller.l_shoulder),
        p1_r_shoulder=float(p1_controller.r_shoulder),
        # Player 2
        p2_button_a=int(p2_controller.button[melee.Button.BUTTON_A]),
        p2_button_b=int(p2_controller.button[melee.Button.BUTTON_B]),
        p2_button_x=int(p2_controller.button[melee.Button.BUTTON_X]),
        p2_button_y=int(p2_controller.button[melee.Button.BUTTON_Y]),
        p2_button_z=int(p2_controller.button[melee.Button.BUTTON_Z]),
        p2_button_start=int(p2_controller.button[melee.Button.BUTTON_START]),
        p2_button_d_up=int(p2_controller.button[melee.Button.BUTTON_D_UP]),
        p2_button_l=int(p2_controller.button[melee.Button.BUTTON_L]),
        p2_button_r=int(p2_controller.button[melee.Button.BUTTON_R]),
        p2_main_stick_x=float(p2_controller.main_stick[0]),
        p2_main_stick_y=float(p2_controller.main_stick[1]),
        p2_c_stick_x=float(p2_controller.c_stick[0]),
        p2_c_stick_y=float(p2_controller.c_stick[1]),
        p2_l_shoulder=float(p2_controller.l_shoulder),
        p2_r_shoulder=float(p2_controller.r_shoulder),
    )

    return gamestate_frame, controller_frame


def process_replay(replay_path: str, min_frames: int = 1500) -> Tuple[FrameData, ...]:
    logger.trace(f"Processing replay {replay_path}")
    try:
        console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        logger.error(f"Error connecting to console for {replay_path}: {e}")
        return tuple()

    frame_data = []
    prev_gamestate_frame: Optional[FrameData] = None
    replay_uuid = hash(replay_path)

    try:
        while True:
            gamestate = console.step()
            if gamestate is None:
                break
            gamestate_frame, controller_frame = extract_single_frame(gamestate, replay_uuid)
            if not gamestate_frame or not controller_frame:
                continue
            # Controller state is stored with resultant gamestate
            # We need to offset by 1 to pair correct input/output for sequential modeling, i.e. what buttons to press next *given the current frame*
            if prev_gamestate_frame is not None:
                prev_gamestate_frame.update(controller_frame)
                frame_data.append(prev_gamestate_frame)
            prev_gamestate_frame = gamestate_frame
    except Exception as e:
        logger.debug(f"Error processing replay {replay_path}: {e}")
    finally:
        console.stop()

    if len(frame_data) < min_frames:
        logger.trace(f"Replay {replay_path} was less than {min_frames} frames, skipping.")
        return tuple()

    return tuple(frame_data)


def write_dataset_incrementally(replay_paths: Tuple[str, ...], output_path: str, batch_size: int) -> None:
    logger.info(f"Processing {len(replay_paths)} replays and writing to {Path(output_path).resolve()}")
    batch = []
    frames_processed = 0

    with mp.Pool() as pool:
        data_generator = pool.imap(process_replay, replay_paths)
        with pq.ParquetWriter(output_path, schema=SCHEMA) as writer:
            pbar = tqdm(data_generator, total=len(replay_paths), desc="Processing replays")
            for replay_data in pbar:
                batch.extend(replay_data)
                frames_processed += len(replay_data)

                if len(batch) >= batch_size:
                    try:
                        table = pa.Table.from_pylist(batch, schema=SCHEMA)
                        writer.write_table(table)
                    except ValueError as e:
                        logger.error(f"Error writing batch: {e}")
                    batch = []
                    pbar.set_description(f"Processed {frames_processed} frames")

            if batch:
                table = pa.Table.from_pylist([attr.asdict(frame) for frame in batch], schema=SCHEMA)
                writer.write_table(table)

    logger.info(f"Finished processing. Total frames: {frames_processed}")


def split_train_val_test(
    input_paths: Tuple[str, ...], train_split: float = 0.9, val_split: float = 0.05, test_split: float = 0.05
) -> dict[str, Tuple[str, ...]]:
    assert train_split + val_split + test_split == 1.0
    n = len(input_paths)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    return {
        "train": tuple(input_paths[:train_end]),
        "val": tuple(input_paths[train_end:val_end]),
        "test": tuple(input_paths[val_end:]),
    }


def process_replays(replay_dir: str, output_dir: str, seed: int, batch_size: int, max_replays: int = -1) -> None:
    replay_paths = list(str(path) for path in Path(replay_dir).rglob("*.slp"))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]
    random.seed(seed)
    random.shuffle(replay_paths)
    splits = split_train_val_test(input_paths=tuple(replay_paths))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split, split_replay_paths in splits.items():
        split_output_path = Path(output_dir) / f"{split}.parquet"
        write_dataset_incrementally(
            replay_paths=split_replay_paths, output_path=str(split_output_path), batch_size=batch_size
        )


def validate_input(replay_dir: str, batch_size: int) -> None:
    if not Path(replay_dir).exists():
        raise ValueError(f"Replay directory does not exist: {replay_dir}")

    if batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer, got: {batch_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Melee replay files and store frame data in parquet.")
    parser.add_argument("--replay_dir", required=True, help="Input directory containing .slp replay files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch", type=int, default=100, help="Number of replay files to process in each batch")
    parser.add_argument("--max_replays", type=int, default=-1, help="Maximum number of replays to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="TRACE")

    validate_input(replay_dir=args.replay_dir, batch_size=args.batch)
    process_replays(
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch,
        max_replays=args.max_replays,
    )
