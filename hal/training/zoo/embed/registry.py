from typing import Callable
from typing import Dict


class Embed:
    EMBED: Dict[str, Callable] = {}

    @classmethod
    def get(cls, name: str) -> Callable:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f"Embedding fn {name} not found." f"Valid functions: {sorted(cls.EMBED.keys())}.")

    @classmethod
    def register(cls, name: str, embed_fn: Callable):
        cls.EMBED[name] = embed_fn
        return embed_fn
