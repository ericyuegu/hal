import argparse
import itertools
import multiprocessing as mp
import random
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import melee
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from more_itertools import chunked
from tqdm import tqdm

from hal.data.constants import IDX_BY_ACTION
from hal.data.constants import IDX_BY_CHARACTER
from hal.data.constants import IDX_BY_STAGE
from hal.data.schema import SCHEMA

FrameData = Dict[str, Any]
ControllerData = Dict[str, Any]


def extract_single_frame(gamestate: melee.GameState, replay_uuid: int) -> Tuple[FrameData, ControllerData]:
    """Extracts gamestate and controller data from one frame of replay."""
    players = sorted(gamestate.players.items())
    if len(players) != 2:
        raise ValueError(f"Expected 2 players, got {len(players)}")

    gamestate_frame: FrameData = {
        # Metadata
        "replay_uuid": replay_uuid,
        "frame": gamestate.frame,
        "stage": IDX_BY_STAGE[gamestate.stage],
    }
    controller_frame: ControllerData = {}

    for i, (port, player) in enumerate(players):
        prefix = f"p{i}_"
        gamestate_frame.update(
            {
                f"{prefix}port": port,
                f"{prefix}character": IDX_BY_CHARACTER[player.character],
                f"{prefix}stock": player.stock,
                f"{prefix}facing": int(player.facing),
                f"{prefix}invulnerable": int(player.invulnerable),
                f"{prefix}position_x": float(player.position.x),
                f"{prefix}position_y": float(player.position.y),
                f"{prefix}percent": player.percent,
                f"{prefix}shield_strength": player.shield_strength,
                f"{prefix}jumps_left": player.jumps_left,
                f"{prefix}action": IDX_BY_ACTION[player.action],
                f"{prefix}action_frame": player.action_frame,
                f"{prefix}invulnerability_left": player.invulnerability_left,
                f"{prefix}hitlag_left": player.hitlag_left,
                f"{prefix}hitstun_left": player.hitstun_frames_left,
                f"{prefix}on_ground": int(player.on_ground),
                f"{prefix}speed_air_x_self": player.speed_air_x_self,
                f"{prefix}speed_y_self": player.speed_y_self,
                f"{prefix}speed_x_attack": player.speed_x_attack,
                f"{prefix}speed_y_attack": player.speed_y_attack,
                f"{prefix}speed_ground_x_self": player.speed_ground_x_self,
                f"{prefix}ecb_bottom_x": player.ecb_bottom[0],
                f"{prefix}ecb_bottom_y": player.ecb_bottom[1],
                f"{prefix}ecb_top_x": player.ecb_top[0],
                f"{prefix}ecb_top_y": player.ecb_top[1],
                f"{prefix}ecb_left_x": player.ecb_left[0],
                f"{prefix}ecb_left_y": player.ecb_left[1],
                f"{prefix}ecb_right_x": player.ecb_right[0],
                f"{prefix}ecb_right_y": player.ecb_right[1],
            }
        )
        controller = player.controller_state
        controller_frame.update(
            {
                f"{prefix}button_a": int(controller.button[melee.Button.BUTTON_A]),
                f"{prefix}button_b": int(controller.button[melee.Button.BUTTON_B]),
                f"{prefix}button_x": int(controller.button[melee.Button.BUTTON_X]),
                f"{prefix}button_y": int(controller.button[melee.Button.BUTTON_Y]),
                f"{prefix}button_z": int(controller.button[melee.Button.BUTTON_Z]),
                f"{prefix}button_start": int(controller.button[melee.Button.BUTTON_START]),
                f"{prefix}button_d_up": int(controller.button[melee.Button.BUTTON_D_UP]),
                f"{prefix}button_l": int(controller.button[melee.Button.BUTTON_L]),
                f"{prefix}button_r": int(controller.button[melee.Button.BUTTON_R]),
                f"{prefix}main_stick_x": float(controller.main_stick[0]),
                f"{prefix}main_stick_y": float(controller.main_stick[1]),
                f"{prefix}c_stick_x": float(controller.c_stick[0]),
                f"{prefix}c_stick_y": float(controller.c_stick[1]),
                f"{prefix}l_shoulder": float(controller.l_shoulder),
                f"{prefix}r_shoulder": float(controller.r_shoulder),
            }
        )

    return gamestate_frame, controller_frame


def process_replay(replay_path: str, min_frames: int = 1500) -> Tuple[FrameData, ...]:
    """Processes a single .slp file and returns the frame data."""
    logger.trace(f"Processing replay {replay_path}")
    try:
        console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        logger.debug(f"Error connecting to console for {replay_path}: {e}")
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
    """Processes a list of replay files and writes the extracted frames to rows in a Parquet file."""
    logger.info(f"Processing {len(replay_paths)} replays and writing to {Path(output_path).resolve()}")
    frames_processed = 0

    with mp.Pool() as pool:
        data_generator = pool.imap(process_replay, replay_paths)
        with pq.ParquetWriter(output_path, schema=SCHEMA) as writer:
            for batch in tqdm(
                chunked(data_generator, batch_size), total=len(replay_paths) // batch_size, desc="batches processed"
            ):
                frames = list(itertools.chain.from_iterable(batch))
                if frames:
                    table = pa.Table.from_pylist(frames, schema=SCHEMA)
                    writer.write_table(table)
                    frames_processed += len(frames)

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
    replay_paths = list(Path(replay_dir).rglob("*.slp"))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]
    random.seed(seed)
    random.shuffle(replay_paths)
    splits = split_train_val_test(input_paths=tuple(map(str, replay_paths)))

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
