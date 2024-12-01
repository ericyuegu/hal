from typing import Dict

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

    for player in [ego, opponent]:
        for feature_name in config.player_features:
            preprocess_fn = normalization_fn_by_feature_name[feature_name]
            player_feature_name = f"{player}_{feature_name}"
            processed_features[player_feature_name] = preprocess_fn(
                sample[player_feature_name], stats[player_feature_name]
            )

    non_player_features = [
        feature for feature in config.player_features if feature not in normalization_fn_by_feature_name
    ]
    for feature_name in non_player_features:
        processed_features[feature_name] = preprocess_fn(sample[feature_name], stats[feature_name])

    concatenated_features_by_head_name: Dict[str, torch.Tensor] = {}
    for head_name, feature_names in config.separate_feature_names_by_head.items():
        features_to_concatenate = [processed_features[feature_name] for feature_name in feature_names]
        concatenated_features_by_head_name[head_name] = torch.cat(features_to_concatenate, dim=-1)

    return concatenated_features_by_head_name
