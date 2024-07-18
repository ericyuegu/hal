from typing import Callable
from typing import Dict

import numpy as np

from hal.data.constants import PLAYER_ACTION_FRAME_FEATURES
from hal.data.constants import PLAYER_ECB_FEATURES
from hal.data.constants import PLAYER_INPUT_FEATURES_TO_EMBED
from hal.data.constants import PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE
from hal.data.constants import PLAYER_INPUT_FEATURES_TO_NORMALIZE
from hal.data.constants import PLAYER_POSITION
from hal.data.constants import PLAYER_SPEED_FEATURES
from hal.data.constants import STAGE
from hal.data.stats import FeatureStats

NormalizationFn = Callable[[np.ndarray, FeatureStats], np.ndarray]


def cast_int32(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Cast to int32."""
    return array.astype(np.int32)


def normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Normalize feature [0, 1]."""
    return ((array - stats.min) / (stats.max - stats.min)).astype(np.float32)


def invert_and_normalize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Invert and normalize feature to [0, 1]."""
    return ((stats.max - array) / (stats.max - stats.min)).astype(np.float32)


def standardize(array: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """Standardize feature to mean 0 and std 1."""
    return ((array - stats.mean) / stats.std).astype(np.float32)


def union(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
    """Perform logical OR of two features."""
    return array_1 | array_2


NORMALIZATION_FN_BY_FEATURE: Dict[str, NormalizationFn] = {
    **dict.fromkeys(STAGE, cast_int32),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, cast_int32),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
    **dict.fromkeys(PLAYER_POSITION, standardize),
    **dict.fromkeys(PLAYER_ACTION_FRAME_FEATURES, normalize),
    **dict.fromkeys(PLAYER_SPEED_FEATURES, standardize),
    **dict.fromkeys(PLAYER_ECB_FEATURES, standardize),
}
