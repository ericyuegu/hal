import attr
import numpy as np

from hal.types import Character


# @attr.s(auto_attribs=True, frozen=True)
# class PlayerState:
#     character: np.array
#     pos_x: np.array
#     pos_y: np.array
#     percent: np.array
#     shield: np.array
#     stock: np.array


@attr.s(auto_attribs=True, frozen=True)
class Replay:
    pass
