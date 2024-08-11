from typing import Callable
from typing import Dict
from typing import Literal

import numpy as np

from hal.data.stats import FeatureStats

Player = Literal["p1", "p2"]
TargetPreprocessFn = Callable[[Dict[str, np.ndarray], Player], Dict[str, np.ndarray]]
InputPreprocessFn = Callable[[Dict[str, np.ndarray], int, Player, Dict[str, FeatureStats]], Dict[str, np.ndarray]]


class InputPreprocessRegistry:
    EMBED: Dict[str, InputPreprocessFn] = {}
    NUM_FEATURES: Dict[str, int] = {}

    @classmethod
    def get(cls, name: str) -> InputPreprocessFn:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f"Embedding fn {name} not found." f"Valid functions: {sorted(cls.EMBED.keys())}.")

    @classmethod
    def register(cls, name: str, num_features: int):
        def decorator(embed_fn: InputPreprocessFn):
            cls.EMBED[name] = embed_fn
            cls.NUM_FEATURES[name] = num_features
            return embed_fn

        return decorator

    @classmethod
    def get_num_features(cls, name: str) -> int:
        if name in cls.NUM_FEATURES:
            return cls.NUM_FEATURES[name]
        raise NotImplementedError(f"Embedding fn {name} not found." f"Valid functions: {sorted(cls.EMBED.keys())}.")


class TargetPreprocessRegistry:
    EMBED: Dict[str, TargetPreprocessFn] = {}

    @classmethod
    def get(cls, name: str) -> TargetPreprocessFn:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f"Embedding fn {name} not found." f"Valid functions: {sorted(cls.EMBED.keys())}.")

    @classmethod
    def register(cls, name: str):
        def decorator(embed_fn: TargetPreprocessFn):
            cls.EMBED[name] = embed_fn
            return embed_fn

        return decorator
