import argparse
import multiprocessing as mp
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import melee
import numpy as np
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from hal.data.constants import IDX_BY_ACTION
from hal.data.constants import IDX_BY_CHARACTER
from hal.data.constants import IDX_BY_STAGE
from hal.data.schema import NP_DTYPE_STR_BY_COLUMN
from hal.data.schema import PYARROW_DTYPE_BY_COLUMN

FrameData = DefaultDict[str, List[Any]]


def setup_logger(output_dir: str | Path) -> None:
    logger.add(Path(output_dir) / "process_replays.log", enqueue=True)


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

    return frame_data


def process_replay(replay_path: str) -> Optional[Dict[str, Any]]:
    frame_data: FrameData = defaultdict(list)
    try:
        console = melee.Console(path=replay_path, is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        logger.debug(f"Error connecting to console for {replay_path}: {e}")
        return None

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
        return None
    finally:
        console.stop()

    # Check if frame_data is valid
    if not frame_data:
        logger.debug(f"No data extracted from replay {replay_path}")
        return None

    # Skip replays with less than `min_frames` frames because they are likely incomplete/low-quality
    min_frames = 1500
    if any(len(v) < min_frames for v in frame_data.values()):
        logger.trace(f"Replay {replay_path} was less than {min_frames} frames, skipping.")
        return None
    # Check for damage
    if all(x == 0 for x in frame_data["p1_percent"]) or all(x == 0 for x in frame_data["p2_percent"]):
        logger.trace(f"Replay {replay_path} had no damage, skipping.")
        return None

    sample = {
        key: np.array(frame_data[key], dtype=dtype.to_pandas_dtype())
        for key, dtype in PYARROW_DTYPE_BY_COLUMN.items()
        if key in frame_data
    }
    sample["replay_uuid"] = np.array([replay_uuid] * len(frame_data["frame"]), dtype=np.int64)
    return sample


def process_replays(
    replay_dir: str, output_dir: str, seed: int, max_replays: int = -1, overwrite_existing: bool = True
) -> None:
    replay_paths = list(Path(replay_dir).rglob("*.slp"))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]
    random.seed(seed)
    random.shuffle(replay_paths)
    splits = split_train_val_test(input_paths=tuple(map(str, replay_paths)))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for split, split_replay_paths in splits.items():
        split_output_dir = Path(output_dir) / f"{split}"
        if overwrite_existing and split_output_dir.exists():
            shutil.rmtree(split_output_dir)
        split_output_dir.mkdir(parents=True, exist_ok=True)
        # Write larger shards to disk, data is repetitive so compression helps a lot
        with MDSWriter(
            out=str(split_output_dir), columns=NP_DTYPE_STR_BY_COLUMN, compression="zstd", size_limit=1 << 30
        ) as out:
            with mp.Pool() as pool:
                samples = pool.imap_unordered(process_replay, split_replay_paths)
                for sample in tqdm(samples, total=len(split_replay_paths), desc=f"Processing {split} split"):
                    if sample is not None:
                        out.write(sample)


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


def validate_input(replay_dir: str) -> None:
    if not Path(replay_dir).exists():
        raise ValueError(f"Replay directory does not exist: {replay_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Melee replay files and store frame data in MDS format.")
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
