from typing import Dict
from typing import Tuple

import attr

from hal.training.config import EmbeddingConfig
from hal.training.preprocess.transform import Transformation


@attr.s(auto_attribs=True)
class InputPreprocessConfig:
    """Configuration for preprocessing functions."""

    # Features to preprocess twice, once for each player
    player_features: Tuple[str, ...]

    # Mapping from feature name to normalization function
    normalization_fn_by_feature_name: Dict[str, Transformation]

    # Mapping from feature name to frame offset relative to sampled index
    # e.g. to include controller inputs from prev frame with current frame gamestate, set p1_button_a = -1, etc.
    frame_offsets_by_feature: Dict[str, int]

    # Mapping from head name to features to be fed to that head
    # Usually for int categorical features
    # All unlisted features are concatenated to the default "gamestate" head
    grouped_feature_names_by_head: Dict[str, Tuple[str, ...]]

    # Input dimensions (D,) of concatenated features after preprocessing
    # TensorDict does not support differentiated sizes across keys for the same dimension
    input_shapes_by_head: Dict[str, Tuple[int, ...]]

    # TODO what if we want to add or remove heads?
    def update_input_shapes_with_embedding_config(
        self, embedding_config: EmbeddingConfig
    ) -> Dict[str, Tuple[int, ...]]:
        new_input_shapes_by_head = self.input_shapes_by_head.copy()
        new_input_shapes_by_head.update(
            {
                "stage": (embedding_config.stage_embedding_dim,),
                "ego_character": (embedding_config.character_embedding_dim,),
                "opponent_character": (embedding_config.character_embedding_dim,),
                "ego_action": (embedding_config.action_embedding_dim,),
                "opponent_action": (embedding_config.action_embedding_dim,),
            }
        )
        self.input_shapes_by_head = new_input_shapes_by_head
        return new_input_shapes_by_head
