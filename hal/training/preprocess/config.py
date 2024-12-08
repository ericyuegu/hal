from typing import Dict
from typing import Tuple

import attr

from hal.data.normalize import NormalizationFn
from hal.data.normalize import cast_int32
from hal.data.normalize import invert_and_normalize
from hal.data.normalize import normalize
from hal.data.normalize import standardize
from hal.training.config import EmbeddingConfig


# TODO what if we want to add or remove heads?
def update_input_shapes_with_embedding_config(
    input_shapes_by_head: Dict[str, Tuple[int, ...]], embedding_config: EmbeddingConfig
) -> Dict[str, Tuple[int, ...]]:
    new_input_shapes_by_head = input_shapes_by_head.copy()
    new_input_shapes_by_head.update(
        {
            "stage": (embedding_config.stage_embedding_dim,),
            "ego_character": (embedding_config.character_embedding_dim,),
            "opponent_character": (embedding_config.character_embedding_dim,),
            "ego_action": (embedding_config.action_embedding_dim,),
            "opponent_action": (embedding_config.action_embedding_dim,),
        }
    )
    return new_input_shapes_by_head


@attr.s(auto_attribs=True)
class InputPreprocessConfig:
    """Configuration for preprocessing functions."""

    # Features to preprocess twice, once for each player
    player_features: Tuple[str, ...]

    # Mapping from feature name to normalization function
    normalization_fn_by_feature_name: Dict[str, NormalizationFn]

    # Mapping from feature name to frame offset relative to sampled index
    # e.g. to include controller inputs from prev frame with current frame gamestate, set p1_button_a = -1, etc.
    offsets_by_feature: Dict[str, int]

    # Mapping from head name to features to be fed to that head
    # All unlisted features are fed to the default "gamestate" head
    separate_feature_names_by_head: Dict[str, Tuple[str, ...]]

    # Input dimensions (D,) of concatenated features after preprocessing
    # TensorDict does not support differentiated sizes across keys for the same dimension
    input_shapes_by_head: Dict[str, Tuple[int, ...]]

    @classmethod
    def v0(cls):
        player_features = (
            "character",
            "action",
            "percent",
            "stock",
            "facing",
            "invulnerable",
            "jumps_left",
            "on_ground",
            "shield_strength",
            "position_x",
            "position_y",
        )

        return cls(
            player_features=player_features,
            normalization_mapping={
                "frame": cast_int32,
                "stage": cast_int32,
                "character": cast_int32,
                "action": cast_int32,
                "percent": normalize,
                "stock": normalize,
                "facing": normalize,
                "invulnerable": normalize,
                "jumps_left": normalize,
                "on_ground": normalize,
                "shield_strength": invert_and_normalize,
                "position_x": standardize,
                "position_y": standardize,
            },
            offsets_by_feature={},
            separate_feature_names_by_head={
                "stage": ("stage",),
                "ego_character": ("ego_character",),
                "opponent_character": ("opponent_character",),
                "ego_action": ("ego_action",),
                "opponent_action": ("opponent_action",),
            },
            input_shapes_by_head={
                "gamestate": (2 * len(player_features),),  # 2x for ego and opponent
            },
        )
