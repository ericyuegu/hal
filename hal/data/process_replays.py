import argparse
import multiprocessing as mp
import random
import shutil
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import melee
import numpy as np
from loguru import logger
from streaming import MDSWriter
from tqdm import tqdm

from hal.data.calculate_stats import calculate_statistics_for_mds
from hal.data.schema import NP_DTYPE_STR_BY_COLUMN
from hal.data.schema import PYARROW_DTYPE_BY_COLUMN
from hal.gamestate_utils import FrameData
from hal.gamestate_utils import extract_and_append_gamestate_inplace


def setup_logger(output_dir: str | Path) -> None:
    logger.add(Path(output_dir) / "process_replays.log", enqueue=True)


def process_replay(replay_path: Path, check_damage: bool = True) -> Optional[Dict[str, Any]]:
    frame_data: FrameData = defaultdict(list)
    try:
        console = melee.Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        logger.debug(f"Error connecting to console for {replay_path}: {e}")
        return None

    replay_uuid = hash(replay_path)

    # Skip first frame since we need both current and next states
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
    if check_damage:
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
    replay_dir: str,
    output_dir: str,
    seed: int,
    max_replays: int = -1,
    max_parallelism: int = 32,
    overwrite_existing: bool = True,
    check_damage: bool = True,
) -> None:
    replay_paths = list(Path(replay_dir).rglob("*.slp"))
    if max_replays > 0:
        replay_paths = replay_paths[:max_replays]
    random.seed(seed)
    random.shuffle(replay_paths)
    splits = split_train_val_test(input_paths=tuple(replay_paths))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    process_replay_partial = partial(process_replay, check_damage=check_damage)

    for split, split_replay_paths in splits.items():
        split_output_dir = Path(output_dir) / f"{split}"
        if overwrite_existing and split_output_dir.exists():
            shutil.rmtree(split_output_dir)
        split_output_dir.mkdir(parents=True, exist_ok=True)
        # Write larger shards to disk, data is repetitive so compression helps a lot
        with MDSWriter(
            out=str(split_output_dir), columns=NP_DTYPE_STR_BY_COLUMN, compression="zstd", size_limit=1 << 30
        ) as out:
            with mp.Pool(max_parallelism) as pool:
                samples = pool.imap_unordered(process_replay_partial, split_replay_paths)
                for sample in tqdm(samples, total=len(split_replay_paths), desc=f"Processing {split} split"):
                    if sample is not None:
                        out.write(sample)


def split_train_val_test(
    input_paths: Tuple[Path, ...], train_split: float = 0.9, val_split: float = 0.05, test_split: float = 0.05
) -> dict[str, Tuple[Path, ...]]:
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
    parser.add_argument(
        "--max_parallelism", type=int, default=32, help="Maximum number of workers to process replays in parallel"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--disable_check_damage", action="store_true", help="Disable damage check in replays")
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
        max_parallelism=args.max_parallelism,
        check_damage=not args.disable_check_damage,
    )

    # Calculate stats
    stats_path = Path(args.output_dir) / "stats.json"
    calculate_statistics_for_mds(args.output_dir, str(stats_path))
