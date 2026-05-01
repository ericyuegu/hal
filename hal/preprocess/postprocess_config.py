from collections.abc import Callable
from typing import Any

import attr
from tensordict import TensorDict


@attr.s(auto_attribs=True)
class PostprocessConfig:
    """Configuration for how we convert model predictions to controller inputs."""

    transformation_by_controller_input: dict[str, Callable[[TensorDict], Any]]
