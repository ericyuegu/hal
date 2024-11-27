import random
from functools import partial
from typing import Dict
from typing import Tuple

import torch
from tensordict import TensorDict

from hal.constants import PLAYER_INPUT_FEATURES_TO_EMBED
from hal.constants import PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE
from hal.constants import PLAYER_INPUT_FEATURES_TO_NORMALIZE
from hal.constants import PLAYER_POSITION
from hal.constants import Player
from hal.constants import STAGE
from hal.constants import VALID_PLAYERS
from hal.constants import get_opponent
from hal.data.normalize import NormalizationFn
from hal.data.normalize import cast_int32
from hal.data.normalize import invert_and_normalize
from hal.data.normalize import normalize
from hal.data.normalize import normalize_and_embed_fourier
from hal.data.normalize import standardize
from hal.data.stats import FeatureStats
from hal.data.stats import load_dataset_stats
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.preprocess.preprocess_targets import preprocess_targets_v0
from hal.training.preprocess.registry import InputPreprocessRegistry
from hal.training.preprocess.registry import PredPostprocessingRegistry
from hal.training.preprocess.registry import TargetPreprocessRegistry


def _preprocess_numeric_features(
    sample: TensorDict,
    player_numeric_features_to_process: Tuple[str, ...],
    ego: Player,
    stats: Dict[str, FeatureStats],
    normalization_fn_by_feature_name: Dict[str, NormalizationFn],
) -> torch.Tensor:
    """Preprocess numeric (gamestate) features for both players."""
    opponent = get_opponent(ego)

    numeric_inputs = []
    for player in [ego, opponent]:
        for feature in player_numeric_features_to_process:
            preprocess_fn: NormalizationFn = normalization_fn_by_feature_name[feature]
            feature_name = f"{player}_{feature}"
            processed_feature = preprocess_fn(sample[feature_name], stats[feature_name])
            if processed_feature.ndim == 1:
                processed_feature = processed_feature.unsqueeze(-1)
            numeric_inputs.append(processed_feature)

    return torch.cat(numeric_inputs, dim=-1)


def _preprocess_categorical_features(
    sample: TensorDict,
    ego: Player,
    stats: Dict[str, FeatureStats],
    normalization_fn_by_feature_name: Dict[str, NormalizationFn],
) -> Dict[str, torch.Tensor]:
    """Preprocess categorical features for both players."""
    opponent = get_opponent(ego)

    def process_feature(feature_name: str, column_name: str) -> torch.Tensor:
        preprocess_fn: NormalizationFn = normalization_fn_by_feature_name[feature_name]
        return preprocess_fn(sample[column_name], stats[column_name]).unsqueeze(-1)

    processed_features = {}

    for feature in PLAYER_INPUT_FEATURES_TO_EMBED:
        for player, prefix in [(ego, "ego"), (opponent, "opponent")]:
            col_name = f"{player}_{feature}"  # e.g. "p1_character"
            perspective_feature_name = f"{prefix}_{feature}"  # e.g. "ego_character"
            processed_features[perspective_feature_name] = process_feature(feature, col_name)

    for feature in STAGE:
        processed_features[feature] = process_feature(feature, column_name=feature)

    return processed_features


def _preprocess_features_by_mapping(
    sample: TensorDict,
    ego: Player,
    stats: Dict[str, FeatureStats],
    player_numeric_feature_names: Tuple[str, ...],
    normalization_fn_by_feature_name: Dict[str, NormalizationFn],
    batch_size: Tuple[int, ...],
) -> TensorDict:
    assert ego in VALID_PLAYERS

    categorical_features = _preprocess_categorical_features(
        sample=sample,
        ego=ego,
        stats=stats,
        normalization_fn_by_feature_name=normalization_fn_by_feature_name,
    )
    gamestate = _preprocess_numeric_features(
        sample=sample,
        player_numeric_features_to_process=player_numeric_feature_names,
        ego=ego,
        stats=stats,
        normalization_fn_by_feature_name=normalization_fn_by_feature_name,
    )

    return TensorDict({**categorical_features, "gamestate": gamestate}, batch_size=batch_size)


PLAYER_NUMERIC_FEATURES_V0 = tuple(
    PLAYER_INPUT_FEATURES_TO_NORMALIZE + PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE + PLAYER_POSITION
)


@InputPreprocessRegistry.register("inputs_v0", num_features=2 * len(PLAYER_NUMERIC_FEATURES_V0))
def preprocess_inputs_v0(
    sample: TensorDict, data_config: DataConfig, ego: Player, stats: Dict[str, FeatureStats]
) -> TensorDict:
    """Slice input sample to the input length.

    Expects tensordict with shape (trajectory_len,)
    """
    trajectory_len = data_config.input_len + data_config.target_len

    player_numeric_feature_names = PLAYER_NUMERIC_FEATURES_V0
    normalization_fn_by_feature_name: Dict[str, NormalizationFn] = {
        **dict.fromkeys(STAGE, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
        **dict.fromkeys(PLAYER_POSITION, standardize),
    }

    return _preprocess_features_by_mapping(
        sample=sample[:trajectory_len],
        ego=ego,
        stats=stats,
        player_numeric_feature_names=player_numeric_feature_names,
        normalization_fn_by_feature_name=normalization_fn_by_feature_name,
        batch_size=(trajectory_len,),
    )


NUMERIC_FEATURES_V1 = tuple(
    PLAYER_INPUT_FEATURES_TO_NORMALIZE + PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE + PLAYER_POSITION
)


# extra input dimensions from Fourier embedding
@InputPreprocessRegistry.register("inputs_v1", num_features=2 * (len(NUMERIC_FEATURES_V1) + 7 * len(PLAYER_POSITION)))
def preprocess_inputs_v1(
    sample: TensorDict, data_config: DataConfig, ego: Player, stats: Dict[str, FeatureStats]
) -> TensorDict:
    """Slice input sample to the input length."""
    trajectory_len = data_config.input_len + data_config.target_len

    numeric_features = NUMERIC_FEATURES_V1
    normalization_fn_by_feature_name = {
        **dict.fromkeys(STAGE, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
        **dict.fromkeys(PLAYER_POSITION, partial(normalize_and_embed_fourier, dim=8)),
    }

    return _preprocess_features_by_mapping(
        sample=sample[:trajectory_len],
        ego=ego,
        stats=stats,
        player_numeric_feature_names=numeric_features,
        normalization_fn_by_feature_name=normalization_fn_by_feature_name,
        batch_size=(trajectory_len,),
    )


@InputPreprocessRegistry.register("inputs_v2", num_features=2 * len(PLAYER_NUMERIC_FEATURES_V0) + 48)
def preprocess_inputs_v2(
    sample: TensorDict, data_config: DataConfig, ego: Player, stats: Dict[str, FeatureStats]
) -> TensorDict:
    player_numeric_feature_names = PLAYER_NUMERIC_FEATURES_V0
    normalization_fn_by_feature_name: Dict[str, NormalizationFn] = {
        **dict.fromkeys(STAGE, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
        **dict.fromkeys(PLAYER_POSITION, standardize),
    }

    # check if sequence length is >1
    if sample.shape[-1] > 1:
        preprocessed_inputs = _preprocess_features_by_mapping(
            sample=sample[1 : data_config.input_len + 1],
            ego=ego,
            stats=stats,
            player_numeric_feature_names=player_numeric_feature_names,
            normalization_fn_by_feature_name=normalization_fn_by_feature_name,
            batch_size=(data_config.input_len,),
        )
        ego_controller = preprocess_targets_v0(sample=sample[0 : data_config.input_len], player=ego)
        preprocessed_inputs.update(ego_controller)
    else:
        preprocessed_inputs = _preprocess_features_by_mapping(
            sample=sample,
            ego=ego,
            stats=stats,
            player_numeric_feature_names=player_numeric_feature_names,
            normalization_fn_by_feature_name=normalization_fn_by_feature_name,
            batch_size=(data_config.input_len,),
        )

    return preprocessed_inputs


"""
TODO:

Create preprocessor class and preprocessing fn registry
- class holds on to data config and knows: 
    - how long example seq len should be
    - how to slice full episodes into appropriate input/target lens for dataset `__getitem__()`
        - e.g. single frame ahead, warmup frames, prev frame for controller inputs
    - numeric input shape
    - categorical input keys & shapes
- single preprocessor fn registry
    - allows distributed registering from multiple files to single registry
- interface:
    - preprocess_inputs
    - preprocess_targets
    - postprocess_preds
    - numeric_input_shape
    - categorical_input_shapes
"""


class Preprocessor:
    def __init__(self, data_config: DataConfig, embedding_config: EmbeddingConfig) -> None:
        self.data_config = data_config
        self.embedding_config = embedding_config
        self.stats = load_dataset_stats(data_config.stats_path)
        self.normalization_fn_by_feature_name: Dict[str, NormalizationFn] = {}

        self.input_len = data_config.input_len
        self.target_len = data_config.target_len

        self.preprocess_inputs_fn = InputPreprocessRegistry.get(embedding_config.input_preprocessing_fn)
        self.preprocess_targets_fn = TargetPreprocessRegistry.get(embedding_config.target_preprocessing_fn)
        self.postprocess_preds_fn = PredPostprocessingRegistry.get(embedding_config.pred_postprocessing_fn)

    @property
    def numeric_input_shape(self) -> int:
        """Get the size of the materialized input dimensions from the embedding config."""
        return InputPreprocessRegistry.get_num_features(self.embedding_config.input_preprocessing_fn)

    @property
    def categorical_input_shapes(self) -> Dict[str, int]:
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

    def sample_supervised_example_from_episode(self, episode_L: TensorDict) -> TensorDict:
        """Randomly slice full .slp episode tensordict into appropriate input/target lengths for supervised training.

        Args:
            episode_L: TensorDict of shape (episode_len,) containing full episode data

        Returns:
            TensorDict of shape (sequence_len,) containing sliced data
        """
        if len(episode_L.shape) != 1:
            raise ValueError(f"Expected episode tensordict with 1 dimension, got shape {episode_L.shape}")

        if self.seq_len > episode_L.shape[0]:
            raise ValueError(f"Episode length {episode_L.shape[0]} is shorter than required length {self.seq_len}")

        start_idx = random.randint(0, episode_L.shape[0] - self.seq_len)

        return episode_L[start_idx : start_idx + self.seq_len]

    def preprocess_inputs(self, sample_L: TensorDict, ego: Player) -> TensorDict:
        return self.preprocess_inputs_fn(sample_L, self.data_config, ego, self.stats)

    def preprocess_targets(self, sample_L: TensorDict, ego: Player) -> TensorDict:
        return self.preprocess_targets_fn(sample_L, ego)

    def postprocess_preds(self, preds_C: TensorDict) -> TensorDict:
        return self.postprocess_preds_fn(preds_C)
