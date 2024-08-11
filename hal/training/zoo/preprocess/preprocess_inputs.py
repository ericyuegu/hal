from typing import Dict
from typing import Tuple

import numpy as np

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


def _preprocess_numeric_features(
    sample: Dict[str, np.ndarray],
    features_to_process: Tuple[str, ...],
    player: str,
    opponent: str,
    stats: Dict[str, FeatureStats],
) -> np.ndarray:
    """Preprocess numeric features for both players."""
    numeric_inputs = []
    for feature in features_to_process:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE_V0[feature]
        for p in [player, opponent]:
            feature_name = f"{p}_{feature}"
            numeric_inputs.append(preprocess_fn(sample[feature_name], stats[feature_name]))  # pylint: disable=E1102
    return np.stack(numeric_inputs, axis=-1)


def _preprocess_categorical_features(
    sample: Dict[str, np.ndarray], player: Player, opponent: Player, stats: Dict[str, FeatureStats]
) -> Dict[str, np.ndarray]:
    """Preprocess categorical features for both players."""

    def process_feature(feature_name: str, column_name: str) -> np.ndarray:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE_V0[feature_name]
        return preprocess_fn(sample[column_name], stats[column_name])[..., np.newaxis]

    processed_features = {}

    for feature in PLAYER_INPUT_FEATURES_TO_EMBED:
        for p, prefix in [(player, "ego"), (opponent, "opponent")]:
            col_name = f"{p}_{feature}"  # e.g. "p1_character"
            perspective_feature_name = f"{prefix}_{feature}"  # e.g. "ego_character"
            processed_features[perspective_feature_name] = process_feature(feature, col_name)

    for feature in STAGE:
        processed_features[feature] = process_feature(feature, column_name=feature)

    return processed_features


@InputPreprocessRegistry.register("inputs_v0", num_features=2 * len(NUMERIC_FEATURES_V0))
def preprocess_inputs_v0(
    sample: Dict[str, np.ndarray], input_len: int, player: Player, stats: Dict[str, FeatureStats]
) -> Dict[str, np.ndarray]:
    """Slice input sample to the input length."""
    assert player in VALID_PLAYERS
    opponent = "p2" if player == "p1" else "p1"

    input_sample = {k: v[:input_len] for k, v in sample.items()}

    categorical_features = _preprocess_categorical_features(input_sample, player, opponent, stats)
    gamestate = _preprocess_numeric_features(input_sample, NUMERIC_FEATURES_V0, player, opponent, stats)

    return {"gamestate": gamestate, **categorical_features}
