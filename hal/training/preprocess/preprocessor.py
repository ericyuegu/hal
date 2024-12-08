import random
from typing import Dict
from typing import Set

import numpy as np
import torch
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import get_opponent
from hal.data.normalize import NormalizationFn
from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.preprocess.config import InputPreprocessConfig
from hal.training.preprocess.config import update_input_shapes_with_embedding_config
from hal.training.preprocess.preprocess_inputs import preprocess_inputs_v2
from hal.training.preprocess.registry import InputPreprocessRegistry


def preprocess_input_features(
    sample: TensorDict,
    ego: Player,
    config: InputPreprocessConfig,
    stats: Dict[str, FeatureStats],
) -> Dict[str, torch.Tensor]:
    """Preprocess input features for a given sample."""
    opponent = get_opponent(ego)
    normalization_fn_by_feature_name = config.normalization_fn_by_feature_name
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


class Preprocessor:
    """
    Container object that converts ndarray dicts of gamestate features into supervised training examples.

    Class holds on to data config and knows:
    - how to slice full episodes into appropriate input/target shapes
        - e.g. single frame ahead, warmup frames, prev frame for controller inputs
    - how long training example seq_len should be
    - input shapes by head
    """

    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        self.data_config = data_config
        self.embedding_config = embedding_config
        self.stats = load_dataset_stats(data_config.stats_path)
        self.normalization_fn_by_feature_name: Dict[str, NormalizationFn] = {}
        self.seq_len = data_config.seq_len

        self.input_preprocess_config = InputPreprocessRegistry.get(self.embedding_config.input_preprocessing_fn)
        self.input_shapes_by_head = update_input_shapes_with_embedding_config(
            self.input_preprocess_config.input_shapes_by_head, self.embedding_config
        )

        # Closed loop eval
        # self.last_controller_inputs: Optional[Dict[str, torch.Tensor]] = None

    @property
    def trajectory_sampling_len(self) -> int:
        """Get the number of frames needed from a full episode to preprocess a supervised training example."""
        trajectory_len = self.seq_len

        # Handle preprocessing fns that require +1 prev frame for controller inputs
        if self.input_preprocess_config in (preprocess_inputs_v2,):
            trajectory_len += 1
        # Other conditions here

        return trajectory_len

    def sample_from_episode(self, ndarrays_by_feature: dict[str, np.ndarray]) -> TensorDict:
        """Randomly slice episode features into input/target sequences for supervised training.

        Args:
            ndarrays_by_feature: dict of shape (episode_len,) containing full episode data

        Returns:
            dict of shape (sequence_len,) containing sliced data
        """
        frames = ndarrays_by_feature["frame"]
        assert all(len(ndarray) == len(frames) for ndarray in ndarrays_by_feature.values())
        episode_len = len(frames)
        sample_index = random.randint(0, episode_len - self.trajectory_sampling_len)
        tensor_slice_by_feature_name = {
            feature_name: torch.from_numpy(
                feature_L[sample_index : sample_index + self.trajectory_sampling_len].copy()
            )
            for feature_name, feature_L in ndarrays_by_feature.items()
        }
        return TensorDict(tensor_slice_by_feature_name, batch_size=(self.trajectory_sampling_len,))

    def preprocess_inputs(self, sample_L: TensorDict, ego: Player) -> TensorDict:
        return self.input_preprocess_config(sample_L, self.data_config, ego, self.stats)

    def preprocess_targets(self, sample_L: TensorDict, ego: Player) -> TensorDict:
        return self.preprocess_targets_fn(sample_L, ego)

    def postprocess_preds(self, preds_C: TensorDict) -> TensorDict:
        return self.postprocess_preds_fn(preds_C)
