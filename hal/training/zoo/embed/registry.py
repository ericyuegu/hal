from typing import Callable
from typing import Dict

import numpy as np

PreprocessFn = Callable[..., Dict[str, np.ndarray]]


class Embed:
    EMBED: Dict[str, PreprocessFn] = {}

    @classmethod
    def get(cls, name: str) -> PreprocessFn:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f"Embedding fn {name} not found." f"Valid functions: {sorted(cls.EMBED.keys())}.")

    @classmethod
    def register(cls, name: str, embed_fn: PreprocessFn):
        cls.EMBED[name] = embed_fn
        return embed_fn
