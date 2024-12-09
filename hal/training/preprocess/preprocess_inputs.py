from typing import Dict
from typing import Set

import torch
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import get_opponent
from hal.data.stats import FeatureStats
from hal.training.preprocess.config import InputPreprocessConfig
from hal.training.preprocess.registry import InputPreprocessRegistry
from hal.training.preprocess.transform import cast_int32
from hal.training.preprocess.transform import invert_and_normalize
from hal.training.preprocess.transform import normalize
from hal.training.preprocess.transform import standardize


def inputs_v0() -> InputPreprocessConfig:
    """
    Baseline input features.

    Separate embedding heads for stage, character, & action.
    No controller, no platforms, no projectiles.
    """

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

    return InputPreprocessConfig(
        player_features=player_features,
        normalization_fn_by_feature_name={
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
        frame_offsets_by_feature={},
        grouped_feature_names_by_head={
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


InputPreprocessRegistry.register("inputs_v0", inputs_v0())


def preprocess_input_features(
    sample: TensorDict,
    ego: Player,
    config: InputPreprocessConfig,
    stats: Dict[str, FeatureStats],
) -> TensorDict:
    """Applies preprocessing functions to player and non-player input features for a given sample.

    Does not slice or shift any features.
    """
    opponent = get_opponent(ego)
    normalization_fn_by_feature_name = config.normalization_fn_by_feature_name
    processed_features: Dict[str, torch.Tensor] = {}

    # Process player features
    for player in (ego, opponent):
        perspective = "ego" if player == ego else "opponent"
        for feature_name in config.player_features:
            # Convert feature name from "p1" to "ego"/"opponent"
            perspective_feature_name = f"{perspective}_{feature_name}"
            player_feature_name = f"{player}_{feature_name}"
            preprocess_fn = normalization_fn_by_feature_name[feature_name]
            processed_features[perspective_feature_name] = preprocess_fn(
                sample[player_feature_name], stats[player_feature_name]
            )

    # Process non-player features
    non_player_features = [
        feature_name for feature_name in normalization_fn_by_feature_name if feature_name not in config.player_features
    ]
    for feature_name in non_player_features:
        processed_features[feature_name] = preprocess_fn(sample[feature_name], stats[feature_name])

    # Concatenate processed features by head
    concatenated_features_by_head_name: Dict[str, torch.Tensor] = {}
    seen_feature_names: Set[str] = set()
    for head_name, feature_names in config.grouped_feature_names_by_head.items():
        features_to_concatenate = [processed_features[feature_name] for feature_name in feature_names]
        concatenated_features_by_head_name[head_name] = torch.cat(features_to_concatenate, dim=-1)
        seen_feature_names.update(feature_names)

    # Add features that are not associated with any head to default `gamestate` head
    DEFAULT_HEAD_NAME = "gamestate"
    unseen_feature_tensors = []
    for feature_name, feature_tensor in processed_features.items():
        if feature_name not in seen_feature_names:
            unseen_feature_tensors.append(feature_tensor)
    concatenated_features_by_head_name[DEFAULT_HEAD_NAME] = torch.cat(unseen_feature_tensors, dim=-1)

    return TensorDict(concatenated_features_by_head_name, batch_size=sample.batch_size)
