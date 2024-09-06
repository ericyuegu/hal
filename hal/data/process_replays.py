import argparse
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import List
from typing import Optional
from typing import Tuple

import melee
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

from hal.data.constants import IDX_BY_ACTION
from hal.data.constants import IDX_BY_CHARACTER
from hal.data.constants import IDX_BY_STAGE
from hal.data.schema import SCHEMA

FrameData = DefaultDict[str, List[Any]]


def setup_logger(output_dir: str | Path) -> None:
    logger.add(Path(output_dir) / "process_replays.log", enqueue=True)


@logger.catch
def extract_and_append_single_frame_inplace(
    frame_data: FrameData, prev_gamestate: Optional[melee.GameState], gamestate: melee.GameState, replay_uuid: int
) -> FrameData:
    """
    Extract gamestate and controller data across two frames of replay and append in-place to `frame_data`.

    Controller state is stored in .slp replays with the resultant gamestate after sending that controller input.
    We need to extract prev gamestate to pair correct input/output for sequential modeling, i.e. what buttons to press next *given the current frame*.
    """
    if prev_gamestate is None:
        return frame_data

    players = sorted(prev_gamestate.players.items())
    if len(players) != 2:
        raise ValueError(f"Expected 2 players, got {len(players)}")

    frame_data["replay_uuid"].append(replay_uuid)
    frame_data["frame"].append(prev_gamestate.frame)
    frame_data["stage"].append(IDX_BY_STAGE[prev_gamestate.stage])

    for i, (port, player_state) in enumerate(players, start=1):
        prefix = f"p{i}_"

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
            frame_data[f"{prefix}{key}"].append(value)

        # Controller data (from current gamestate)
        controller = gamestate.players[port].controller_state

        # Button data
        buttons = ["A", "B", "X", "Y", "Z", "START", "L", "R", "D_UP"]
        for button in buttons:
            frame_data[f"{prefix}button_{button.lower()}"].append(
                int(controller.button[getattr(melee.Button, f"BUTTON_{button}")])
            )

        # Stick and shoulder data
        frame_data[f"{prefix}main_stick_x"].append(float(controller.main_stick[0]))
        frame_data[f"{prefix}main_stick_y"].append(float(controller.main_stick[1]))
        frame_data[f"{prefix}c_stick_x"].append(float(controller.c_stick[0]))
        frame_data[f"{prefix}c_stick_y"].append(float(controller.c_stick[1]))
        frame_data[f"{prefix}l_shoulder"].append(float(controller.l_shoulder))
        frame_data[f"{prefix}r_shoulder"].append(float(controller.r_shoulder))

    return frame_data


@logger.catch
def process_replay(replay_path: str, min_frames: int = 1500) -> FrameData:
    """Processes a single .slp file and returns the frame data."""
    logger.trace(f"Processing replay {replay_path}")

    frame_data: FrameData = defaultdict(list)
    try:
        console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        logger.debug(f"Error connecting to console for {replay_path}: {e}")
        return frame_data

    replay_uuid = hash(replay_path)
    prev_gamestate: Optional[melee.GameState] = None

    try:
        while True:
            gamestate = console.step()
            if gamestate is None:
                break
            frame_data = extract_and_append_single_frame_inplace(
                frame_data=frame_data, prev_gamestate=prev_gamestate, gamestate=gamestate, replay_uuid=replay_uuid
            )
            prev_gamestate = gamestate
    except Exception as e:
        logger.debug(f"Error processing replay {replay_path}: {e}")
    finally:
        console.stop()

    # Assert lists aren't empty
    for key, value in frame_data.items():
        assert len(value) > 0, f"List for key {key} is empty"
    # Skip replays with less than `min_frames` frames because they are likely incomplete/low-quality
    if any(len(v) < min_frames for v in frame_data.values()):
        logger.trace(f"Replay {replay_path} was less than {min_frames} frames, skipping.")
        return defaultdict(list)
    # Check for damage
    if all(x == 0 for x in frame_data["p1_percent"]) or all(x == 0 for x in frame_data["p2_percent"]):
        logger.trace(f"Replay {replay_path} had no damage, skipping.")
        return defaultdict(list)

    return frame_data


@logger.catch
def write_dataset_incrementally(replay_paths: Tuple[str, ...], output_path: str, worker_init_fn: Callable) -> None:
    logger.info(f"Processing {len(replay_paths)} replays and writing to {Path(output_path).resolve()}")
    frames_processed = 0

    t0 = time.perf_counter()
    with mp.Pool(initializer=worker_init_fn) as pool:
        data_generator = pool.imap(process_replay, replay_paths)
        pbar = tqdm(data_generator, total=len(replay_paths))
        with pq.ParquetWriter(output_path, schema=SCHEMA) as writer:
            for frame_data in pbar:
                if frame_data:
                    table = pa.Table.from_pydict(frame_data, schema=SCHEMA)
                    writer.write_table(table)
                    frames_processed += len(next(iter(frame_data.values())))
                    pbar.set_description(f"Replays processed: {pbar.n}/{len(replay_paths)}")

    t1 = time.perf_counter()
    logger.info(f"Finished processing {len(replay_paths)} replays in {t1 - t0:.2f} seconds.")


def split_train_val_test(
    input_paths: Tuple[str, ...], train_split: float = 0.9, val_split: float = 0.05, test_split: float = 0.05
) -> dict[str, Tuple[str, ...]]:
    assert train_split + val_split + test_split == 1.0
    n = len(input_paths)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)
    return {
        "val": tuple(input_paths[train_end:val_end]),
        "test": tuple(input_paths[val_end:]),
        "train": tuple(input_paths[:train_end]),
    }


def process_replays(replay_dir: str, output_dir: str, seed: int, max_replays: int = -1) -> None:
    replay_paths = list(Path(replay_dir).rglob("*.slp"))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]
    random.seed(seed)
    random.shuffle(replay_paths)
    splits = split_train_val_test(input_paths=tuple(map(str, replay_paths)))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split, split_replay_paths in splits.items():
        split_output_path = Path(output_dir) / f"{split}.parquet"
        worker_init_fn = partial(setup_logger, output_dir=output_dir)
        write_dataset_incrementally(
            replay_paths=split_replay_paths, output_path=str(split_output_path), worker_init_fn=worker_init_fn
        )


def validate_input(replay_dir: str) -> None:
    if not Path(replay_dir).exists():
        raise ValueError(f"Replay directory does not exist: {replay_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Melee replay files and store frame data in parquet.")
    parser.add_argument("--replay_dir", required=True, help="Input directory containing .slp replay files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_replays", type=int, default=-1, help="Maximum number of replays to process")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    setup_logger(output_dir=args.output_dir)

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="TRACE")

    validate_input(replay_dir=args.replay_dir)
    process_replays(
        replay_dir=args.replay_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_replays=args.max_replays,
    )
