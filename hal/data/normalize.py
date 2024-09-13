from typing import Callable
from typing import TypeVar

import numpy as np
import torch

from hal.data.stats import FeatureStats

ArrayLike = TypeVar("ArrayLike", np.ndarray, torch.Tensor)
NormalizationFn = Callable[[ArrayLike, FeatureStats], ArrayLike]


def cast_int32(array: ArrayLike, stats: FeatureStats):
    """Cast to int32."""
    if isinstance(array, np.ndarray):
        return array.astype(np.int32)
    elif isinstance(array, torch.Tensor):
        return array.to(torch.int32)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")


def normalize(array: ArrayLike, stats: FeatureStats):
    """Normalize feature [0, 1]."""
    if isinstance(array, np.ndarray):
        return ((array - stats.min) / (stats.max - stats.min)).astype(np.float32)
    elif isinstance(array, torch.Tensor):
        return ((array - stats.min) / (stats.max - stats.min)).to(torch.float32)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")


def invert_and_normalize(array: ArrayLike, stats: FeatureStats):
    """Invert and normalize feature to [0, 1]."""
    if isinstance(array, np.ndarray):
        return ((stats.max - array) / (stats.max - stats.min)).astype(np.float32)
    elif isinstance(array, torch.Tensor):
        return ((stats.max - array) / (stats.max - stats.min)).to(torch.float32)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")


def standardize(array: ArrayLike, stats: FeatureStats):
    """Standardize feature to mean 0 and std 1."""
    if isinstance(array, np.ndarray):
        return ((array - stats.mean) / stats.std).astype(np.float32)
    elif isinstance(array, torch.Tensor):
        return ((array - stats.mean) / stats.std).to(torch.float32)
    else:
        raise TypeError("Input should be a numpy array or a torch tensor")


def union(array_1: ArrayLike, array_2: ArrayLike):
    """Perform logical OR of two features."""
    if isinstance(array_1, np.ndarray) and isinstance(array_2, np.ndarray):
        return array_1 | array_2
    elif isinstance(array_1, torch.Tensor) and isinstance(array_2, torch.Tensor):
        return array_1 | array_2
    else:
        raise TypeError("Inputs should be both numpy arrays or both torch tensors")


def normalize_and_embed_fourier(array: ArrayLike, stats: FeatureStats, dim: int = 8):
    """Normalize then embed values at various frequencies."""
    normalized = normalize(array, stats)
    frequencies = 1024 * torch.linspace(0, -torch.tensor(10000.).log(), dim // 2).exp()
    emb = normalized.view(-1, 1) * frequencies
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
