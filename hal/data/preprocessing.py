# %%
import time
from typing import Dict

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats

np.set_printoptions(threshold=np.inf)


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


# def one_hot_3d(arr: np.ndarray) -> np.ndarray:
#     """One hot encode 3d array."""
#     # Identify the start of each streak
#     streak = np.diff(arr, axis=-1, prepend=0) > 0
#     result = streak * arr

#     # Set last column to 1 if row is empty
#     result[..., -1] = ~np.any(arr, axis=-1)

#     return result


# def one_hot_3d(arr):
#     B, T, D = arr.shape

#     # Step 1: Identify the newest streak
#     diff = np.diff(arr, axis=-2, prepend=0)
#     newest_streak = diff == 1

#     # Step 2: Create a mask for the newest streak in each row
#     row_has_newest = newest_streak.any(axis=-1)
#     newest_index = np.where(row_has_newest, newest_streak.argmax(axis=2), -1)
#     mask = np.zeros_like(arr, dtype=bool)
#     mask[np.arange(B)[:, None], np.arange(T), newest_index] = True

#     # Step 3: Apply the mask to keep only the newest streak
#     result = np.where(mask, arr, 0)

#     # Step 4: Zero mask, set last column to 1
#     result[..., -1] = ~np.any(arr, axis=-1)

#     return result


def one_hot_3d(arr):
    B, T, D = arr.shape

    # Step 1: Identify the newest streak
    diff = np.diff(arr, axis=-2, prepend=0)
    newest_streak = diff == 1
    print(f"{newest_streak=}")

    # Step 2: Create a cumulative sum to propagate the newest streak
    cumsum = np.cumsum(newest_streak, axis=-2)
    print(f"{cumsum=}")

    # Step 3: Create a mask for the newest streak in each row
    row_max = np.maximum.accumulate(cumsum, axis=-1)
    print(f"{row_max=}")
    mask = (cumsum == row_max) & (arr == 1)
    print(f"{mask=}")

    # Step 4: Apply the mask to keep only the newest streak
    result = np.where(mask, arr, 0)
    print(f"{result=}")

    return result


feature_processors = {
    INPUT_FEATURES_TO_EMBED: lambda x: x,
    INPUT_FEATURES_TO_NORMALIZE: normalize,
    INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: invert_and_normalize,
    INPUT_FEATURES_TO_STANDARDIZE: standardize,
}


START, END = 885, 900


def preprocess_features_v0(sample: Dict[str, np.ndarray], stats: Dict[str, FeatureStats]) -> Dict[str, np.ndarray]:
    """Preprocess features."""
    preprocessed = {}

    # Stack buttons and encode one_hot
    for player in ("p1", "p2"):
        button_a = (sample[f"{player}_button_a"]).astype(np.bool_)
        button_b = (sample[f"{player}_button_b"]).astype(np.bool_)
        button_z = sample[f"{player}_button_z"]
        jump = union(sample[f"{player}_button_x"], sample[f"{player}_button_y"])
        shoulder = union(sample[f"{player}_button_l"], sample[f"{player}_button_r"])

        stacked_buttons = np.stack((button_a, button_b, button_z, jump, shoulder), axis=1)[np.newaxis, ...]
        if player == "p1":
            print(stacked_buttons[0, START:END])
        preprocessed[f"{player}_buttons"] = one_hot_3d(stacked_buttons)

    # for feature_list, preprocessing_func in feature_processors.items():
    #     for feature in feature_list:
    #         process_feature(feature, preprocessing_func)

    return preprocessed


input_path = "/opt/projects/hal2/data/dev/val.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"

table: pa.Table = pq.read_table(input_path, memory_map=True)
stats = load_dataset_stats(stats_path)

table_slice = pyarrow_table_to_np_dict(table)
player = "p1"

button_a = (table_slice[f"{player}_button_a"]).astype(np.bool_)
button_b = (table_slice[f"{player}_button_b"]).astype(np.bool_)
button_z = table_slice[f"{player}_button_z"]
jump = union(table_slice[f"{player}_button_x"], table_slice[f"{player}_button_y"])
shoulder = union(table_slice[f"{player}_button_l"], table_slice[f"{player}_button_r"])
no_button = np.zeros_like(button_a)


arr1 = np.array(
    [
        [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    ],
    dtype=np.int8,
)
print(arr1)
print(one_hot_3d(arr1))

# %%

stacked_buttons = np.stack((button_a, button_b, button_z, jump, shoulder, no_button), axis=1)[np.newaxis, ...]
print(stacked_buttons.shape)
print(stacked_buttons[:, START:END])

t0 = time.perf_counter()
preprocessed = one_hot_3d(stacked_buttons)
t1 = time.perf_counter()
# print(f"Time to preprocess features: {t1 - t0} seconds")
print(preprocessed.shape)
print(preprocessed[:, START:END])

# %%
print(arr1)
print(stacked_buttons[:, START:END])
