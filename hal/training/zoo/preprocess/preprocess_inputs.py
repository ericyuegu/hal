from typing import Dict
from typing import Tuple

import numpy as np
import torch
from tensordict import TensorDict

from hal.data.constants import PLAYER_INPUT_FEATURES_TO_EMBED
from hal.data.constants import PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE
from hal.data.constants import PLAYER_INPUT_FEATURES_TO_NORMALIZE
from hal.data.constants import PLAYER_POSITION
from hal.data.constants import STAGE
from hal.data.constants import VALID_PLAYERS
from hal.data.normalize import NormalizationFn
from hal.data.normalize import cast_int32
from hal.data.normalize import invert_and_normalize
from hal.data.normalize import normalize
from hal.data.normalize import standardize
from hal.data.stats import FeatureStats
from hal.training.config import DataConfig
from hal.training.zoo.preprocess.registry import InputPreprocessRegistry
from hal.training.zoo.preprocess.registry import Player

NORMALIZATION_FN_BY_FEATURE_V0: Dict[str, NormalizationFn] = {
    **dict.fromkeys(STAGE, cast_int32),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, cast_int32),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
    **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
    **dict.fromkeys(PLAYER_POSITION, standardize),
    # **dict.fromkeys(PLAYER_ACTION_FRAME_FEATURES, normalize),
    # **dict.fromkeys(PLAYER_SPEED_FEATURES, standardize),
    # **dict.fromkeys(PLAYER_ECB_FEATURES, standardize),
}

NUMERIC_FEATURES_V0 = tuple(
    PLAYER_INPUT_FEATURES_TO_NORMALIZE + PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE + PLAYER_POSITION
)


def _get_opponent(player: Player) -> Player:
    return "p2" if player == "p1" else "p1"


def _preprocess_numeric_features(
    sample: TensorDict,
    features_to_process: Tuple[str, ...],
    ego: Player,
    stats: Dict[str, FeatureStats],
) -> torch.Tensor:
    """Preprocess numeric (gamestate) features for both players."""
    opponent = _get_opponent(ego)

    numeric_inputs = []
    for player in [ego, opponent]:
        for feature in features_to_process:
            preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE_V0[feature]
            feature_name = f"{player}_{feature}"
            numeric_inputs.append(preprocess_fn(sample[feature_name], stats[feature_name]))

    return torch.stack(numeric_inputs, dim=-1)


def _preprocess_categorical_features(
    sample: TensorDict, ego: Player, stats: Dict[str, FeatureStats]
) -> Dict[str, torch.Tensor]:
    """Preprocess categorical features for both players."""
    opponent = _get_opponent(ego)

    def process_feature(feature_name: str, column_name: str) -> np.ndarray:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE_V0[feature_name]
        return preprocess_fn(sample[column_name], stats[column_name])[..., np.newaxis]

    processed_features = {}

    for feature in PLAYER_INPUT_FEATURES_TO_EMBED:
        for player, prefix in [(ego, "ego"), (opponent, "opponent")]:
            col_name = f"{player}_{feature}"  # e.g. "p1_character"
            perspective_feature_name = f"{prefix}_{feature}"  # e.g. "ego_character"
            processed_features[perspective_feature_name] = process_feature(feature, col_name)

    for feature in STAGE:
        processed_features[feature] = process_feature(feature, column_name=feature)

    return processed_features


@InputPreprocessRegistry.register("inputs_v0", num_features=2 * len(NUMERIC_FEATURES_V0))
def preprocess_inputs_v0(
    sample: TensorDict, data_config: DataConfig, ego: Player, stats: Dict[str, FeatureStats]
) -> TensorDict:
    """Slice input sample to the input length."""
    assert ego in VALID_PLAYERS
    trajectory_len = data_config.input_len + data_config.target_len

    categorical_features = _preprocess_categorical_features(sample[:trajectory_len], ego=ego, stats=stats)
    gamestate = _preprocess_numeric_features(
        sample=sample[:trajectory_len], features_to_process=NUMERIC_FEATURES_V0, ego=ego, stats=stats
    )

    categorical_features["gamestate"] = gamestate
    return TensorDict(categorical_features, batch_size=(trajectory_len,))
