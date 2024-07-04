import argparse
import random
from pathlib import Path
from typing import Tuple

import attr
import melee
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from loguru import logger

from hal.data.primitives import FrameData
from hal.data.primitives import SCHEMA


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
        p1_ecb_bottom_x=p1.ecb_bottom[0],
        p1_ecb_bottom_y=p1.ecb_bottom[1],
        p1_ecb_top_x=p1.ecb_top[0],
        p1_ecb_top_y=p1.ecb_top[1],
        p1_ecb_left_x=p1.ecb_left[0],
        p1_ecb_left_y=p1.ecb_left[1],
        p1_ecb_right_x=p1.ecb_right[0],
        p1_ecb_right_y=p1.ecb_right[1],
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
        p2_ecb_bottom_x=p2.ecb_bottom[0],
        p2_ecb_bottom_y=p2.ecb_bottom[1],
        p2_ecb_top_x=p2.ecb_top[0],
        p2_ecb_top_y=p2.ecb_top[1],
        p2_ecb_left_x=p2.ecb_left[0],
        p2_ecb_left_y=p2.ecb_left[1],
        p2_ecb_right_x=p2.ecb_right[0],
        p2_ecb_right_y=p2.ecb_right[1],
    )


def process_replay(replay_path: str) -> Tuple[FrameData, ...]:
    console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
    try:
        console.connect()
    except Exception as e:
        logger.error(f"Error connecting to console: {e}")
        return tuple()

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

    return tuple(frame_data)


def write_dataset_incrementally(replay_paths: Tuple[Path, ...], output_path: str, batch_size: int = 1000) -> None:
    batch = []
    frames_processed = 0

    part = ds.partitioning(pa.schema([SCHEMA.field("stage")]))

    with pq.ParquetWriter(output_path, schema=SCHEMA, partitioning=part) as writer:
        for replay_path in replay_paths:
            replay_data = process_replay(str(replay_path))
            batch.extend(replay_data)
            frames_processed += len(replay_data)

            if len(batch) >= batch_size:
                table = pa.Table.from_pylist([attr.asdict(frame) for frame in batch], schema=SCHEMA)
                writer.write_table(table)

                batch = []
                logger.info(f"Processed {frames_processed} frames")

        if batch:
            table = pa.Table.from_pylist([attr.asdict(frame) for frame in batch], schema=SCHEMA)
            writer.write_table(table)

    logger.info(f"Finished processing. Total frames: {frames_processed}")


def split_train_val_test(
    input_paths: Tuple[Path, ...], train_split: float = 0.9, val_split: float = 0.05, test_split: float = 0.05
) -> dict[str, Tuple[Path, ...]]:
    assert train_split + val_split + test_split == 1.0
    n = len(input_paths)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    return {
        "train": tuple(input_paths[:train_end]),
        "val": tuple(input_paths[train_end:val_end]),
        "test": tuple(input_paths[val_end:]),
    }


def process_replays(replay_dir: str, output_dir: str, seed: int, batch_size: int = 100) -> None:
    random.seed(seed)
    replay_paths = list(Path(replay_dir).rglob("*.slp"))
    random.shuffle(replay_paths)

    splits = split_train_val_test(input_paths=tuple(replay_paths))
    for split, split_paths in splits.items():
        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        write_dataset_incrementally(replay_paths=split_paths, output_path=str(split_dir), batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Melee replay files and store frame data in parquet.")
    parser.add_argument("--replay_dir", required=True, help="Input directory containing .slp replay files")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of replay files to process in each batch")
    args = parser.parse_args()

    process_replays(args.replay_dir, args.output_dir, args.seed, args.batch_size)
