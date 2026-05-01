from collections.abc import Callable
from typing import Any

import torch


class Arch:
    # Model constructor and params
    ARCH: dict[str, tuple[Callable[..., torch.nn.Module], dict[str, Any]]] = {}

    @classmethod
    def get(cls, name: str, **kwargs) -> torch.nn.Module:
        if name in cls.ARCH:
            model_class, model_params = cls.ARCH[name]
            return model_class(**model_params, **kwargs)
        raise NotImplementedError(f"Architecture {name} not found.Valid architectures: {sorted(cls.ARCH.keys())}.")

    @classmethod
    def register(cls, name: str, make_net: Callable[..., torch.nn.Module], **kwargs) -> Callable[..., torch.nn.Module]:
        cls.ARCH[name] = make_net, kwargs
        return make_net
