from typing import Callable
from typing import Dict
from typing import Final
from typing import Tuple

import numpy as np
import pyarrow as pa

from hal.data.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.data.stats import FeatureStats

VALID_PLAYERS: Final[Tuple[str, str]] = ("p1", "p2")


###################
# Normalization   #
###################


STAGE: Tuple[str, ...] = ("stage",)
PLAYER_INPUT_FEATURES_TO_EMBED: Tuple[str, ...] = ("character", "action")
PLAYER_INPUT_FEATURES_TO_NORMALIZE: Tuple[str, ...] = (
    "percent",
    "stock",
    "facing",
    "action_frame",
    "invulnerable",
    "jumps_left",
    "on_ground",
)
PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: Tuple[str, ...] = ("shield_strength",)
PLAYER_POSITION: Tuple[str, ...] = (
    "position_x",
    "position_y",
)
# Optional input features
PLAYER_HITLAG_FEATURES: Tuple[str, ...] = (
    "hitlag_left",
    "hitstun_left",
)
PLAYER_SPEED_FEATURES: Tuple[str, ...] = (
    "speed_air_x_self",
    "speed_y_self",
    "speed_x_attack",
    "speed_y_attack",
    "speed_ground_x_self",
)
PLAYER_ECB_FEATURES: Tuple[str, ...] = (
    "ecb_bottom_x",
    "ecb_bottom_y",
    "ecb_top_x",
    "ecb_top_y",
    "ecb_left_x",
    "ecb_left_y",
    "ecb_right_x",
    "ecb_right_y",
)
# Target features
TARGET_FEATURES_TO_ONE_HOT_ENCODE: Tuple[str, ...] = (
    "button_a",
    "button_b",
    "button_x",
    "button_z",
    "button_l",
)


NormalizationFn = Callable[[np.ndarray, FeatureStats], np.ndarray]


def identity(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Identity function."""
    return array


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


NORMALIZATION_FN_BY_FEATURE: Dict[str, NormalizationFn] = {
    **dict.fromkeys(STAGE, identity),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, identity),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
    **dict.fromkeys(PLAYER_POSITION, standardize),
    **dict.fromkeys(PLAYER_HITLAG_FEATURES, normalize),
    **dict.fromkeys(PLAYER_SPEED_FEATURES, standardize),
    **dict.fromkeys(PLAYER_ECB_FEATURES, standardize),
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
