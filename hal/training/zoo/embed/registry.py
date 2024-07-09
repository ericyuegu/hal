from typing import Callable
from typing import Dict

import attr
import numpy as np
import numpy.typing as npt


@attr.s(auto_attribs=True, frozen=True)
class ModelInputs:
    stage: npt.NDArray[np.int_]
    ego_character: npt.NDArray[np.int_]
    ego_action: npt.NDArray[np.int_]
    opponent_character: npt.NDArray[np.int_]
    opponent_action: npt.NDArray[np.int_]
    gamestate: npt.NDArray[np.float32]


@attr.s(auto_attribs=True, frozen=True)
class ModelTargets:
    main_stick: npt.NDArray[np.int_]
    c_stick: npt.NDArray[np.int_]
    buttons: npt.NDArray[np.int_]


InputPreprocessFn = Callable[..., ModelInputs]
TargetPreprocessFn = Callable[..., ModelTargets]


class InputPreprocessRegistry:
    EMBED: Dict[str, InputPreprocessFn] = {}

    @classmethod
    def get(cls, name: str) -> InputPreprocessFn:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f"Embedding fn {name} not found." f"Valid functions: {sorted(cls.EMBED.keys())}.")

    @classmethod
    def register(cls, name: str):
        def decorator(embed_fn: InputPreprocessFn):
            cls.EMBED[name] = embed_fn
            return embed_fn

        return decorator


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
