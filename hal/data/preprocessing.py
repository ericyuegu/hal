# %%
from typing import Dict

import numpy as np
import pyarrow as pa
from data.constants import STICK_XY_CLUSTER_CENTERS_V0

from hal.data.stats import FeatureStats

###################
# Normalization   #
###################

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


PREPROCESS_FN_BY_FEATURE = {
    **dict.fromkeys(INPUT_FEATURES_TO_EMBED, lambda x: x),
    **dict.fromkeys(INPUT_FEATURES_TO_NORMALIZE, normalize),
    **dict.fromkeys(INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
    **dict.fromkeys(INPUT_FEATURES_TO_STANDARDIZE, standardize),
}


###################
# Encoding        #
###################


def one_hot_3d_fast_bugged(arr: np.ndarray) -> np.ndarray:
    """
    One-hot encode 3d array of raw button presses.

    Vectorized but slightly wrong; this takes the left-most 1 in each row instead of the newest button press, so buttons have a cardinal order of priority.

    Sets the last button to 1 if there are no 1s in the array.

    Args:
        arr (np.ndarray): Input array of shape (B, T, D) where B is the batch size,
                          T is the number of time steps, and D is the number of buttons + 1.

    Returns:
        np.ndarray: One-hot encoded array of the same shape (B, T, D).
    """
    # Find where 1s start in each row
    start_mask = np.concatenate([arr[:, :, 0:1], arr[:, :, 1:] > arr[:, :, :-1]], axis=2)
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


def one_hot_2d_fast(arr: np.ndarray) -> np.ndarray:
    """
    One-hot encode 2D array of raw button presses.

    Vectorized but slightly wrong; this takes the left-most 1 in each row instead of the newest button press, so buttons have a cardinal order of priority.

    Sets the last button to 1 if there are no 1s in the array.

    Args:
        arr (np.ndarray): Input array of shape (T, D) where T is the number of time steps
                          and D is the number of buttons.

    Returns:
        np.ndarray: One-hot encoded array of the same shape (T, D).
    """
    # Find where 1s start in each row
    start_mask = np.concatenate([arr[:, 0:1], arr[:, 1:] > arr[:, :-1]], axis=-1)
    # Compute cumulative sum to identify streaks
    cumsum = np.cumsum(start_mask, axis=-1)
    streak_ids = cumsum * arr
    # Find the maximum streak ID for each row
    max_streak_ids = np.max(streak_ids, axis=-1, keepdims=True)
    # Create a mask for the old streak in each row
    old_streak_mask = (streak_ids == max_streak_ids) & (streak_ids > 1) & (arr == 1)
    # Process the array
    processed = arr * ~old_streak_mask
    # Handle rows with no 1s
    no_ones_mask = ~np.any(arr, axis=-1)
    processed[no_ones_mask, -1] = 1

    return processed


def get_closest_stick_xy_cluster_v0(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the closest point in STICK_XY_CLUSTER_CENTERS_V0 for given x and y values.

    Args:
        x (np.ndarray): (T,) X-coordinates in range [0, 1]
        y (np.ndarray): (T,) Y-coordinates in range [0, 1]

    Returns:
        np.ndarray: (T,) Indices of the closest cluster centers
    """
    point = np.stack((x, y), axis=-1)  # Shape: (T, 2)
    distances = np.sum((STICK_XY_CLUSTER_CENTERS_V0 - point[:, np.newaxis, :]) ** 2, axis=-1)
    return np.argmin(distances, axis=-1)


###################
# Reshaping       #
###################


def pyarrow_table_to_np_dict(table: pa.Table) -> Dict[str, np.ndarray]:
    """Convert pyarrow table to dictionary of numpy arrays."""
    return {name: col.to_numpy() for name, col in zip(table.column_names, table.columns)}


def preprocess_target_v0(sample: Dict[str, np.ndarray], player: str) -> Dict[str, np.ndarray]:
    """Return one-hot encoded buttons and analog stick values for given player."""
    target = {}

    # Main stick and c-stick classification
    main_stick_x = sample[f"{player}_main_stick_x"]
    main_stick_y = sample[f"{player}_main_stick_y"]
    c_stick_x = sample[f"{player}_c_stick_x"]
    c_stick_y = sample[f"{player}_c_stick_y"]
    main_stick_clusters = get_closest_stick_xy_cluster_v0(main_stick_x, main_stick_y)
    c_stick_clusters = get_closest_stick_xy_cluster_v0(c_stick_x, c_stick_y)
    target["main_stick"] = main_stick_clusters
    target["c_stick"] = c_stick_clusters

    # Stack buttons and encode one_hot
    button_a = sample[f"{player}_button_a"]
    button_b = sample[f"{player}_button_b"]
    jump = union(sample[f"{player}_button_x"], sample[f"{player}_button_y"])
    button_z = sample[f"{player}_button_z"]
    shoulder = union(sample[f"{player}_button_l"], sample[f"{player}_button_r"])
    stacked_buttons = np.stack((button_a, button_b, jump, button_z, shoulder), axis=1)[np.newaxis, ...]
    target["buttons"] = one_hot_3d_fast_bugged(stacked_buttons)

    return target


# %%
import pyarrow as pa
from pyarrow import parquet as pq

input_path = "/opt/projects/hal2/data/dev/val.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"

table: pa.Table = pq.read_table(input_path, memory_map=True)

replay = table[:1000]
player = "p1"
sample = pyarrow_table_to_np_dict(replay)
# batch_size = 4
# x_batches = np.stack(np.array_split(sample["p1_main_stick_x"], batch_size, axis=0))
# y_batches = np.stack(np.array_split(sample["p1_main_stick_y"], batch_size, axis=0))
x = sample["p1_main_stick_x"]
y = sample["p1_main_stick_y"]
clusters = get_closest_stick_xy_cluster_v0(x, y)
# %%
x, y
# %%
clusters
