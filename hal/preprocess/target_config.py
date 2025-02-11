from typing import Dict
from typing import Tuple

import attr

from hal.preprocess.transformations import Transformation


@attr.s(auto_attribs=True)
class TargetConfig:
    """Configuration for how we structure input features, offsets, and grouping into heads."""

    # Controller inputs
    transformation_by_feature: Dict[str, Transformation]

    # Mapping from feature name to frame offset relative to sampled index
    # e.g. to predict controller inputs from 5 frames in the future, set buttons_5 = 5, etc.
    # +1 HAS ALREADY BEEN APPLIED TO CONTROLLER INPUTS AT DATASET CREATION,
    # meaning next frame's controller ("targets") are matched with current frame's gamestate ("inputs")
    frame_offsets_by_target: Dict[str, int]

    # Input dimensions (D,) of concatenated features after preprocessing
    # TensorDict does not support differentiated sizes across keys for the same dimension
    target_shapes_by_head: Dict[str, Tuple[int, ...]]

    @property
    def target_size(self) -> int:
        """Total dimension of target features."""
        return sum(shape[0] for shape in self.target_shapes_by_head.values())
