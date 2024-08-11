from typing import Callable

import numpy as np

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
