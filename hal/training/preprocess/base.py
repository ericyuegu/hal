from typing import Dict
from typing import Set

import torch
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import get_opponent
from hal.data.stats import FeatureStats
from hal.training.preprocess.config import InputPreprocessConfig


def preprocess_input_features(
    sample: TensorDict,
    ego: Player,
    config: InputPreprocessConfig,
    stats: Dict[str, FeatureStats],
) -> Dict[str, torch.Tensor]:
    opponent = get_opponent(ego)
    normalization_fn_by_feature_name = config.normalization_mapping
    processed_features: Dict[str, torch.Tensor] = {}

    # Process player features
    for player in [ego, opponent]:
        for feature_name in config.player_features:
            preprocess_fn = normalization_fn_by_feature_name[feature_name]
            player_feature_name = f"{player}_{feature_name}"
            processed_features[player_feature_name] = preprocess_fn(
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
    for head_name, feature_names in config.separate_feature_names_by_head.items():
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

    return concatenated_features_by_head_name
