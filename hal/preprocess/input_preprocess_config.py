from typing import Dict
from typing import Tuple

import attr

from hal.preprocess.transform import Transformation
from hal.training.config import DataConfig


@attr.s(auto_attribs=True)
class InputPreprocessConfig:
    """Configuration for preprocessing functions."""

    # Features to preprocess twice, specific to player state
    player_features: Tuple[str, ...]

    # Mapping from feature name to normalization function
    # Must include embedded features such as stage, character, action, but embedding happens at model arch
    normalization_fn_by_feature_name: Dict[str, Transformation]

    # Mapping from feature name to frame offset relative to sampled index
    # e.g. to include controller inputs from prev frame with current frame gamestate, set p1_button_a = -1, etc.
    # +1 HAS ALREADY BEEN APPLIED TO CONTROLLER INPUTS AT DATASET CREATION,
    # meaning next frame's controller ("targets") are matched with current frame's gamestate ("inputs")
    frame_offsets_by_feature: Dict[str, int]

    # Mapping from head name to features to be fed to that head
    # Usually for int categorical features
    # All unlisted features are concatenated to the default "gamestate" head
    grouped_feature_names_by_head: Dict[str, Tuple[str, ...]]

    # Input dimensions (D,) of concatenated features after preprocessing
    # TensorDict does not support differentiated sizes across keys for the same dimension
    input_shapes_by_head: Dict[str, Tuple[int, ...]]

    # TODO what if we want to add or remove heads?
    def update_input_shapes_with_data_config(self, data_config: DataConfig) -> Dict[str, Tuple[int, ...]]:
        new_input_shapes_by_head = self.input_shapes_by_head.copy()
        new_input_shapes_by_head.update(
            {
                "stage": (data_config.stage_embedding_dim,),
                "ego_character": (data_config.character_embedding_dim,),
                "opponent_character": (data_config.character_embedding_dim,),
                "ego_action": (data_config.action_embedding_dim,),
                "opponent_action": (data_config.action_embedding_dim,),
            }
        )
        self.input_shapes_by_head = new_input_shapes_by_head
        return new_input_shapes_by_head
