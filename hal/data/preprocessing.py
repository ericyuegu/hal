# %%
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
)


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


def preprocess_features(table: pa.Table, stats: Dict[str, FeatureStats]) -> pa.Table:
    """Preprocess features."""
    for feature in SCHEMA:
        if feature in table.column_names:
            table = table.drop(feature)
    return table


# %%
# load dataset, load stats and apply them to the dataset
input_path = "/opt/projects/hal2/data/dev/train.parquet"
stats_path = "/opt/projects/hal2/data/dev/stats.json"

table: pa.Table = pq.read_table(input_path)
stats = load_dataset_stats(stats_path)

# %%
shield = table["p1_shield_strength"].to_numpy()
shield = invert_and_normalize(shield, stats["p1_shield_strength"])
shield[:10000]

# %%
shield.max()
