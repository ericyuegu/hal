from typing import Dict, Tuple, Callable, Any

import torch


class Embed:
    EMBED: Dict[str, Callable] = {}

    @classmethod
    def get(cls, name: str) -> Callable:
        if name in cls.EMBED:
            return cls.EMBED[name]
        raise NotImplementedError(f'Embedding fn {name} not found.'
                                  f'Valid functions: {sorted(cls.EMBED.keys())}.')

    @classmethod
    def register(cls, name: str, embed_fn: Callable):
        cls.EMBED[name] = embed_fn
        return embed_fn


class Arch:
    # Model constructor and params
    ARCH: Dict[str, Tuple[Callable[..., torch.nn.Module], Dict[str, Any]]] = {}

    @classmethod
    def get(cls, name: str, **kwargs) -> torch.nn.Module:
        if name in cls.ARCH:
            model_class, model_params = cls.ARCH[name]
            return model_class(**model_params, **kwargs)
        raise NotImplementedError(f'Architecture {name} not found.'
                                  f'Valid architectures: {sorted(cls.ARCH.keys())}.')

    @classmethod
    def register(cls, name: str, make_net: Callable[..., torch.nn.Module], **kwargs) -> Callable[..., torch.nn.Module]:
        cls.ARCH[name] = make_net, kwargs
        return make_net
