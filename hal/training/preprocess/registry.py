from typing import Callable
from typing import Dict
from typing import Literal

from tensordict import TensorDict

from hal.data.stats import FeatureStats
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig

Player = Literal["p1", "p2"]

InputPreprocessFn = Callable[[TensorDict, DataConfig, Player, Dict[str, FeatureStats]], TensorDict]


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


def get_input_size_from_config(config: EmbeddingConfig) -> int:
    """Get the size of the materialized input dimensions from the embedding config."""
    numeric_feature_count = InputPreprocessRegistry.get_num_features(config.input_preprocessing_fn)
    return (
        numeric_feature_count
        + config.stage_embedding_dim
        + 2 * (config.character_embedding_dim + config.action_embedding_dim)
    )
