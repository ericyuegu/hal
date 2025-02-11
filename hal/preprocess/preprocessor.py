import random
from typing import Dict

import numpy as np
import torch
from tensordict import TensorDict

from hal.constants import Player
from hal.data.stats import load_dataset_stats
from hal.preprocess.preprocess_inputs import preprocess_input_features
from hal.preprocess.registry import InputPreprocessRegistry
from hal.preprocess.registry import PredPostprocessingRegistry
from hal.preprocess.registry import TargetPreprocessRegistry
from hal.preprocess.transform import Transformation
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

        self.input_preprocess_config = InputPreprocessRegistry.get(self.data_config.input_preprocessing_fn)
        self.input_shapes_by_head = self.input_preprocess_config.update_input_shapes_with_data_config(self.data_config)
        self.preprocess_targets_fn = TargetPreprocessRegistry.get(self.data_config.target_preprocessing_fn)
        self.postprocess_preds_fn = PredPostprocessingRegistry.get(self.data_config.pred_postprocessing_fn)

        self.frame_offsets_by_feature = self.input_preprocess_config.frame_offsets_by_feature
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
            config=self.input_preprocess_config,
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
