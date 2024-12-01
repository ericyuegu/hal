from typing import Callable
from typing import Dict
from typing import Tuple

from tensordict import TensorDict

from hal.constants import Player
from hal.data.stats import FeatureStats
from hal.training.config import DataConfig
from hal.training.preprocess.config import InputPreprocessConfig

InputPreprocessFn = Callable[[TensorDict, DataConfig, Player, Dict[str, FeatureStats]], TensorDict]


class InputPreprocessRegistry:
    EMBED: Dict[str, InputPreprocessFn] = {}
    CONFIGS: Dict[str, InputPreprocessConfig] = {}

    @classmethod
    def get(cls, name: str) -> InputPreprocessFn:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f"Preprocessing fn {name} not found. Valid functions: {sorted(cls.EMBED.keys())}.")

    @classmethod
    def get_config(cls, name: str) -> InputPreprocessConfig:
        """Get the config class associated with a preprocessing function."""
        return cls.CONFIGS[name]

    @classmethod
    def register(cls, name: str, config: InputPreprocessConfig):
        """Register a preprocessing function with an optional config class."""

        def decorator(preprocess_fn: InputPreprocessFn):
            cls.EMBED[name] = preprocess_fn
            cls.CONFIGS[name] = config
            return preprocess_fn

        return decorator

    @classmethod
    def get_input_sizes(cls, name: str) -> Dict[str, Tuple[int, ...]]:
        """Get input sizes for all heads from a registered config."""
        config_cls = cls.get_config(name)
        return config_cls.input_shapes_by_head


TargetPreprocessFn = Callable[[TensorDict, Player], TensorDict]


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


PredPostprocessFn = Callable[[TensorDict], TensorDict]


class PredPostprocessingRegistry:
    EMBED: Dict[str, PredPostprocessFn] = {}

    @classmethod
    def get(cls, name: str) -> PredPostprocessFn:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f"Embedding fn {name} not found." f"Valid functions: {sorted(cls.EMBED.keys())}.")

    @classmethod
    def register(cls, name: str):
        def decorator(embed_fn: PredPostprocessFn):
            cls.EMBED[name] = embed_fn
            return embed_fn

        return decorator
