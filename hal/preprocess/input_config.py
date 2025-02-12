from typing import Dict
from typing import Tuple

import attr

from hal.preprocess.transformations import Transformation


@attr.s(auto_attribs=True)
class InputConfig:
    """Configuration for how we structure input features, offsets, and grouping into heads."""

    # Features to preprocess twice, specific to player state
    player_features: Tuple[str, ...]

    # Mapping from feature name to transformation function
    # Must include embedded features such as stage, character, action, but embedding happens at model arch
    # Feature names that do not exist in raw sample are assumed to preprocess using multiple features
    transformation_by_feature_name: Dict[str, Transformation]

    # Mapping from transformed/preprocessed input to frame offset relative to sample index
    # e.g. to include controller inputs from prev frame with current frame gamestate, set p1_button_a = -1, etc.
    # +1 HAS ALREADY BEEN APPLIED TO CONTROLLER INPUTS AT DATASET CREATION,
    # meaning next frame's controller ("targets") are matched with current frame's gamestate ("inputs")
    frame_offsets_by_input: Dict[str, int]

    # Mapping from head name to features to be fed to that head
    # Usually for int categorical features
    # All unlisted features are concatenated to the default "gamestate" head
    grouped_feature_names_by_head: Dict[str, Tuple[str, ...]]

    # Input dimensions (D,) of concatenated features after preprocessing
    # TensorDict does not support differentiated sizes across keys for the same dimension
    # *May be dynamically updated with data config*
    input_shapes_by_head: Dict[str, Tuple[int, ...]]

    @property
    def input_size(self) -> int:
        """Total dimension of input features."""
        return sum(shape[0] for shape in self.input_shapes_by_head.values())
