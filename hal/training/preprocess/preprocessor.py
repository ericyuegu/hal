import random
from typing import Dict
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from hal.constants import Player
from hal.data.normalize import NormalizationFn
from hal.data.stats import load_dataset_stats
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.preprocess.preprocess_inputs import preprocess_inputs_v2
from hal.training.preprocess.registry import InputPreprocessRegistry
from hal.training.preprocess.registry import PredPostprocessingRegistry
from hal.training.preprocess.registry import TargetPreprocessRegistry


class Preprocessor:
    """
    Converts ndarray dict of gamestate features into supervised training examples.

    Class holds on to data config and knows:
    - how to slice full episodes into appropriate input/target shapes
        - e.g. single frame ahead, warmup frames, prev frame for controller inputs
    - how long training example seq_len should be
    - numeric input shape
    - categorical input keys & shapes
    """

    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        self.data_config = data_config
        self.embedding_config = embedding_config
        self.stats = load_dataset_stats(data_config.stats_path)
        self.normalization_fn_by_feature_name: Dict[str, NormalizationFn] = {}

        self.input_len = data_config.input_len
        self.target_len = data_config.target_len

        self.preprocess_inputs_fn = InputPreprocessRegistry.get(self.embedding_config.input_preprocessing_fn)
        self.preprocess_targets_fn = TargetPreprocessRegistry.get(self.embedding_config.target_preprocessing_fn)
        self.postprocess_preds_fn = PredPostprocessingRegistry.get(self.embedding_config.pred_postprocessing_fn)

        # Closed loop eval
        self.last_controller_inputs: Optional[Dict[str, torch.Tensor]] = None

    @property
    def numeric_input_shape(self) -> int:
        """Get the size of the materialized input dimensions from the embedding config."""
        return InputPreprocessRegistry.get_num_features(self.embedding_config.input_preprocessing_fn)

    @property
    def categorical_input_shapes(self) -> dict[str, int]:
        return {
            "stage": self.embedding_config.stage_embedding_dim,
            "ego_character": self.embedding_config.character_embedding_dim,
            "opponent_character": self.embedding_config.character_embedding_dim,
            "ego_action": self.embedding_config.action_embedding_dim,
            "opponent_action": self.embedding_config.action_embedding_dim,
        }

    @property
    def trajectory_sampling_len(self) -> int:
        """Get the number of frames needed from a full episode to preprocess a supervised training example."""
        trajectory_len = self.input_len + self.target_len

        # Handle preprocessing fns that require +1 prev frame for controller inputs
        if self.preprocess_inputs_fn in (preprocess_inputs_v2,):
            trajectory_len += 1
        # Other conditions here

        return trajectory_len

    @property
    def seq_len(self) -> int:
        """Get the final length of a preprocessed supervised training example / sequence."""
        return self.input_len + self.target_len

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
        return self.preprocess_inputs_fn(sample_L, self.data_config, ego, self.stats)

    def preprocess_targets(self, sample_L: TensorDict, ego: Player) -> TensorDict:
        return self.preprocess_targets_fn(sample_L, ego)

    def postprocess_preds(self, preds_C: TensorDict) -> TensorDict:
        return self.postprocess_preds_fn(preds_C)
