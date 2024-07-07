from typing import Dict

import numpy as np
import pyarrow as pa

from hal.data.stats import FeatureStats

INPUT_FEATURES_TO_EMBED = ("stage", "character", "action")
INPUT_FEATURES_TO_NORMALIZE = ("percent", "stock", "facing", "action_frame", "invulnerable", "jumps_left", "on_ground")
INPUT_FEATURES_TO_INVERT_AND_NORMALIZE = ("shield_strength",)
INPUT_FEATURES_TO_STANDARDIZE = (
    "position_x",
    "position_y",
    "hitlag_left",
    "hitstun_left",
    "speed_air_x_self",
    "speed_y_self",
    "speed_x_attack",
    "speed_y_attack",
    "speed_ground_x_self",
)

TARGET_FEATURES_TO_ONE_HOT_ENCODE = ("button_a", "button_b", "button_x", "button_z", "button_l")


def pyarrow_table_to_np_dict(table: pa.Table) -> Dict[str, np.ndarray]:
    """Convert pyarrow table to dictionary of numpy arrays."""
    return {name: col.to_numpy() for name, col in zip(table.column_names, table.columns)}


def normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Normalize feature [0, 1]."""
    return (array - stats.min) / (stats.max - stats.min)


def invert_and_normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Invert and normalize feature to [0, 1]."""
    return (stats.max - array) / (stats.max - stats.min)


def standardize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Standardize feature to mean 0 and std 1."""
    return (array - stats.mean) / stats.std


def union(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Perform logical OR of two features."""
    return array_1 | array_2


def one_hot_3d_fast_bugged(arr: np.ndarray) -> np.ndarray:
    """
    One-hot encode 3d array of raw button presses.

    Vectorized but slightly wrong; this takes the left-most 1 in each row instead of the newest button press, so buttons have a cardinal order of priority.
    """
    # Find where 1s start in each row
    start_mask = np.concatenate([arr[:, :, 0:1], arr[:, :, 1:] > arr[:, :, :-1]], axis=2)
    print(f"{start_mask=}")
    # Compute cumulative sum to identify streaks
    cumsum = np.cumsum(start_mask, axis=2)
    streak_ids = cumsum * arr
    # Find the maximum streak ID for each row
    max_streak_ids = np.max(streak_ids, axis=2, keepdims=True)
    # Create a mask for the old streak in each row
    old_streak_mask = (streak_ids == max_streak_ids) & (streak_ids > 1) & (arr == 1)
    processed = arr * ~old_streak_mask
    # Handle rows with no 1s
    no_ones_mask = ~np.any(arr, axis=2)
    processed[no_ones_mask, -1] = 1

    return processed


feature_processors = {
    INPUT_FEATURES_TO_EMBED: lambda x: x,
    INPUT_FEATURES_TO_NORMALIZE: normalize,
    INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: invert_and_normalize,
    INPUT_FEATURES_TO_STANDARDIZE: standardize,
}


def preprocess_target_v0(sample: Dict[str, np.ndarray], player: str) -> Dict[str, np.ndarray]:
    """Return one-hot encoded buttons and analog stick values for given player."""
    target = {}

    # TODO analog sticks

    # Stack buttons and encode one_hot
    button_a = (sample[f"{player}_button_a"]).astype(np.bool_)
    button_b = (sample[f"{player}_button_b"]).astype(np.bool_)
    jump = union(sample[f"{player}_button_x"], sample[f"{player}_button_y"])
    button_z = sample[f"{player}_button_z"]
    shoulder = union(sample[f"{player}_button_l"], sample[f"{player}_button_r"])
    stacked_buttons = np.stack((button_a, button_b, jump, button_z, shoulder), axis=1)[np.newaxis, ...]
    target[f"buttons"] = one_hot_3d_fast_bugged(stacked_buttons)

    return target
