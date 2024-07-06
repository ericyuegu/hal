# %%
from typing import Dict

import numpy as np
import pyarrow as pa
from data.constants import ACTION_BY_IDX
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


def vectorized_custom_one_hot(arr):
    rows, cols = arr.shape

    # Create a matrix of column indices
    col_indices = np.arange(cols).reshape(1, -1).repeat(rows, axis=0)

    # Create a mask for valid positions (where arr == 1)
    valid_mask = arr == 1

    # Handle rows with no 1s
    rows_without_ones = ~np.any(valid_mask, axis=1)
    valid_mask[rows_without_ones, -1] = True

    # Find the rightmost valid position for each row
    rightmost_valid = np.where(valid_mask, col_indices, -1).max(axis=1)

    # Create a matrix of cumulative max of rightmost valid positions
    cummax_rightmost = np.maximum.accumulate(rightmost_valid)

    # Create the final selection mask
    valid_and_in_range = (col_indices <= cummax_rightmost.reshape(-1, 1)) & valid_mask
    rightmost_valid_in_range = np.where(valid_and_in_range, col_indices, -1).max(axis=1)
    selection_mask = col_indices == rightmost_valid_in_range.reshape(-1, 1)

    # Convert the mask to the final result
    result = selection_mask.astype(int)

    return result


feature_processors = {
    INPUT_FEATURES_TO_EMBED: lambda x: x,
    INPUT_FEATURES_TO_NORMALIZE: normalize,
    INPUT_FEATURES_TO_INVERT_AND_NORMALIZE: invert_and_normalize,
    INPUT_FEATURES_TO_STANDARDIZE: standardize,
}


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
        no_button = np.zeros_like(sample[f"{player}_button_a"])

        stacked_buttons = np.stack((button_a, button_b, button_z, jump, shoulder, no_button), axis=1)
        preprocessed[f"{player}_buttons"] = vectorized_custom_one_hot(stacked_buttons)

    # for feature_list, preprocessing_func in feature_processors.items():
    #     for feature in feature_list:
    #         process_feature(feature, preprocessing_func)

    return preprocessed


# %%
# load dataset, load stats and apply them to the dataset
input_path = "/opt/projects/hal2/data/dev/val.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"

table: pa.Table = pq.read_table(input_path, memory_map=True)
stats = load_dataset_stats(stats_path)

# %%
shield = table["p1_shield_strength"].to_numpy()
shield = invert_and_normalize(shield, stats["p1_shield_strength"])
shield[:10000]

# %%
table_slice = table
preprocessed = preprocess_features_v0(pyarrow_table_to_np_dict(table_slice), stats)
# %%
print(preprocessed["p1_buttons"][886:900])

# %%
# find rows where multiple buttons are pressed
buttons = np.stack(
    [
        table["p1_button_a"].to_numpy(),
        table["p1_button_b"].to_numpy(),
        table["p1_button_x"].to_numpy(),
        table["p1_button_y"].to_numpy(),
        table["p1_button_z"].to_numpy(),
        table["p1_button_l"].to_numpy(),
        table["p1_button_r"].to_numpy(),
    ],
    axis=1,
)

multiple_buttons_pressed = np.sum(buttons, axis=1) >= 2
indices = np.where(multiple_buttons_pressed)[0]
for index in indices:
    print(f"Index: {index}, Buttons: {buttons[index]}")

# %%
[ACTION_BY_IDX[i] for i in table[886:900]["p1_action"].to_pylist()]

# %%
buttons[886:900]
# %%

# print(table[multiple_buttons_pressed])
# start, end = 205, 215
# for button in ("a", "b", "z", "x", "y", "l", "r"):
#     print(table_slice[f"p1_button_{button}"].to_numpy()[start:end])

# %%
a = preprocessed["p1_buttons"]
b = table_slice["p1_button_x"].to_numpy() | table_slice["p1_button_y"].to_numpy()
a[:, 3] == b
# %%
