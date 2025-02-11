import random
from typing import Dict
from typing import Set

import attr
import numpy as np
import torch
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import get_opponent
from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats
from hal.preprocess.input_config import InputConfig
from hal.preprocess.input_configs import DEFAULT_HEAD_NAME
from hal.preprocess.registry import InputConfigRegistry
from hal.preprocess.registry import PredPostprocessingRegistry
from hal.preprocess.registry import TargetPreprocessRegistry
from hal.preprocess.transformations import Transformation
from hal.training.config import DataConfig


class Preprocessor:
    """
    Converts ndarray dicts of gamestate features into training examples.

    We support frame offsets for features during supervised training,
    e.g. grouping controller inputs from a previous frame with the current frame's gamestate.

    Class holds on to data config and knows:
    - how to slice full episodes into appropriate input/target shapes
    - how many frames to offset features
        - e.g. warmup frames, prev frame for controller inputs, multiple frames ahead for multi-step predictions
    - hidden dim sizes by input embedding head at runtime
    """

    def __init__(self, data_config: DataConfig) -> None:
        self.data_config = data_config
        self.stats = load_dataset_stats(data_config.stats_path)
        self.normalization_fn_by_feature_name: Dict[str, Transformation] = {}
        self.seq_len = data_config.seq_len

        self.input_config = InputConfigRegistry.get(self.data_config.input_preprocessing_fn)
        # Dynamically update registered config with user-specified embedding shapes
        self.input_config = update_input_shapes_with_data_config(self.input_config, data_config)
        self.preprocess_targets_fn = TargetPreprocessRegistry.get(self.data_config.target_preprocessing_fn)
        self.postprocess_preds_fn = PredPostprocessingRegistry.get(self.data_config.pred_postprocessing_fn)

        self.frame_offsets_by_feature = self.input_config.frame_offsets_by_feature
        self.max_abs_offset = max((abs(offset) for offset in self.frame_offsets_by_feature.values()), default=0)
        self.min_offset = min((offset for offset in self.frame_offsets_by_feature.values()), default=0)

    @property
    def eval_warmup_frames(self) -> int:
        """If min_offset is negative, we need to skip min_offset frames at eval time to match training distribution."""
        if self.min_offset < 0:
            return abs(self.min_offset)
        return 0

    @property
    def trajectory_sampling_len(self) -> int:
        """Calculates number of frames needed from a full episode to preprocess a supervised training example."""
        trajectory_len = self.seq_len
        trajectory_len += self.max_abs_offset
        return trajectory_len

    @property
    def input_size(self) -> int:
        return sum(shape[0] for shape in self.input_shapes_by_head.values())

    def sample_from_episode(self, ndarrays_by_feature: dict[str, np.ndarray], debug: bool = False) -> TensorDict:
        """Randomly slice input/target features into trajectory_sampling_len sequences for supervised training.

        Can be substituted with feature buffer at eval / runtime.

        Args:
            ndarrays_by_feature: dict of shape (episode_len,) containing full episode data

        Returns:
            TensorDict of shape (trajectory_sampling_len,)
        """
        frames = ndarrays_by_feature["frame"]
        assert all(len(ndarray) == len(frames) for ndarray in ndarrays_by_feature.values())
        episode_len = len(frames)
        sample_index = 0 if debug else random.randint(0, episode_len - self.trajectory_sampling_len)
        tensor_slice_by_feature_name = {
            feature_name: torch.from_numpy(
                feature_L[sample_index : sample_index + self.trajectory_sampling_len].copy()
            )
            for feature_name, feature_L in ndarrays_by_feature.items()
        }
        return TensorDict(tensor_slice_by_feature_name, batch_size=(self.trajectory_sampling_len,))

    def offset_features(self, sample_T: TensorDict) -> TensorDict:
        """Offset & slice features to training-ready sequence length.

        Args:
            sample_T: TensorDict of shape (trajectory_sampling_len,) containing features

        Returns:
            TensorDict of shape (seq_len,) with features offset according to config
        """
        # What frame the training sequence starts on
        reference_frame_idx = abs(min(0, self.min_offset))
        offset_features = {}

        for feature_name, tensor in sample_T.items():
            offset = self.frame_offsets_by_feature.get(feature_name, 0)
            # What frame this feature is sampled from / to
            start_idx = reference_frame_idx + offset
            end_idx = start_idx + self.seq_len
            offset_features[feature_name] = tensor[start_idx:end_idx]

        return TensorDict(offset_features, batch_size=(self.seq_len,))

    def preprocess_inputs(self, sample_L: TensorDict, ego: Player) -> TensorDict:
        return preprocess_input_features(
            sample=sample_L,
            ego=ego,
            config=self.input_config,
            stats=self.stats,
        )

    def preprocess_targets(self, sample_L: TensorDict, ego: Player) -> TensorDict:
        return self.preprocess_targets_fn(sample_L, ego)

    def postprocess_preds(self, preds_C: TensorDict) -> TensorDict:
        return self.postprocess_preds_fn(preds_C)

    def mock_preds_as_tensordict(self) -> TensorDict:
        """Mock a single model prediction."""
        out = {
            name: torch.zeros(num_clusters)
            for name, num_clusters in {
                "buttons": self.data_config.num_buttons,
                "main_stick": self.data_config.num_main_stick_clusters,
                "c_stick": self.data_config.num_c_stick_clusters,
                "shoulder": self.data_config.num_shoulder_clusters,
            }.items()
            if num_clusters is not None
        }
        return TensorDict(out, batch_size=())


def preprocess_input_features(
    sample: TensorDict,
    ego: Player,
    config: InputConfig,
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
        preprocess_fn = normalization_fn_by_feature_name[feature_name]
        if feature_name in sample:
            # Single feature transformation
            processed_features[feature_name] = preprocess_fn(sample[feature_name], stats[feature_name])
        else:
            # Multi-feature transformation (e.g. controller inputs)
            # Pass entire dict and player perspective
            processed_features[feature_name] = preprocess_fn(sample, ego)
    # Concatenate processed features by head
    concatenated_features_by_head_name: Dict[str, torch.Tensor] = {}
    seen_feature_names: Set[str] = set()
    for head_name, feature_names in config.grouped_feature_names_by_head.items():
        features_to_concatenate = [processed_features[feature_name] for feature_name in feature_names]
        concatenated_features_by_head_name[head_name] = torch.cat(features_to_concatenate, dim=-1)
        seen_feature_names.update(feature_names)

    # Add features that are not associated with any head to default `gamestate` head
    unseen_feature_tensors = []
    for feature_name, feature_tensor in processed_features.items():
        if feature_name not in seen_feature_names:
            if feature_tensor.ndim == 1:
                feature_tensor = feature_tensor.unsqueeze(-1)
            unseen_feature_tensors.append(feature_tensor)
    concatenated_features_by_head_name[DEFAULT_HEAD_NAME] = torch.cat(unseen_feature_tensors, dim=-1)

    return TensorDict(concatenated_features_by_head_name, batch_size=sample.batch_size)


def update_input_shapes_with_data_config(input_config: InputConfig, data_config: DataConfig) -> InputConfig:
    return attr.evolve(
        input_config,
        input_shapes_by_head={
            **input_config.input_shapes_by_head,
            "stage": (data_config.stage_embedding_dim,),
            "ego_character": (data_config.character_embedding_dim,),
            "opponent_character": (data_config.character_embedding_dim,),
            "ego_action": (data_config.action_embedding_dim,),
            "opponent_action": (data_config.action_embedding_dim,),
        },
    )
