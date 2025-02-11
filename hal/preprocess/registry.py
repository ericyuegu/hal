from typing import Callable
from typing import Dict

from tensordict import TensorDict

from hal.constants import Player
from hal.preprocess.input_config import InputConfig


class InputConfigRegistry:
    CONFIGS: Dict[str, InputConfig] = {}

    @classmethod
    def get(cls, name: str) -> InputConfig:
        if name in cls.CONFIGS:
            return cls.CONFIGS[name]
        raise NotImplementedError(f"Preprocessing fn {name} not found. Valid functions: {sorted(cls.CONFIGS.keys())}.")

    @classmethod
    def register(cls, name: str, config: InputConfig) -> None:
        cls.CONFIGS[name] = config


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
