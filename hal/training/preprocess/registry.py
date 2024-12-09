from typing import Callable
from typing import Dict
from typing import Tuple

from tensordict import TensorDict

from hal.constants import Player
from hal.training.preprocess.preprocess_inputs import InputPreprocessConfig


class InputPreprocessRegistry:
    CONFIGS: Dict[str, InputPreprocessConfig] = {}

    @classmethod
    def get(cls, name: str) -> InputPreprocessConfig:
        if name in cls.CONFIGS:
            return cls.CONFIGS[name]
        raise NotImplementedError(f"Preprocessing fn {name} not found. Valid functions: {sorted(cls.CONFIGS.keys())}.")

    @classmethod
    def register(cls, name: str, config: InputPreprocessConfig):
        if name in cls.CONFIGS:
            raise ValueError(f"InputPreprocessConfig with name '{name}' already registered.")
        cls.CONFIGS[name] = config

    @classmethod
    def get_input_sizes(cls, name: str) -> Dict[str, Tuple[int, ...]]:
        """Get input sizes for all heads from a registered config."""
        config_cls = cls.get(name)
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
