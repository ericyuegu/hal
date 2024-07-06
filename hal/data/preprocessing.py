# %%
from typing import Dict

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats

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


def encode_stacked_array_to_one_hot(array: np.ndarray) -> np.ndarray:
    """Convert stacked features to one hot encoding."""
    # Argmax tiebreaks by returning the first element
    first_one_indices = np.argmax(array, axis=1)
    one_hot = np.zeros_like(array)
    one_hot[np.arange(array.shape[0]), first_one_indices] = 1

    return one_hot


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
        jump = union(sample[f"{player}_button_x"], sample[f"{player}_button_y"])
        button_z = sample[f"{player}_button_z"]
        shoulder = union(sample[f"{player}_button_l"], sample[f"{player}_button_r"])

        stacked_buttons = np.stack((button_a, button_b, button_z, jump, shoulder), axis=1)
        preprocessed[f"{player}_buttons"] = encode_stacked_array_to_one_hot(stacked_buttons)

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
table_slice = table[900:1000]
preprocess_features_v0(pyarrow_table_to_np_dict(table_slice), stats)
# %%
