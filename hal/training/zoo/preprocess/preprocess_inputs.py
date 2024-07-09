from typing import Dict

import numpy as np

from hal.data.normalize import NORMALIZATION_FN_BY_FEATURE
from hal.data.normalize import NormalizationFn
from hal.data.normalize import PLAYER_INPUT_FEATURES_TO_EMBED
from hal.data.normalize import PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE
from hal.data.normalize import PLAYER_INPUT_FEATURES_TO_NORMALIZE
from hal.data.normalize import PLAYER_POSITION
from hal.data.normalize import VALID_PLAYERS
from hal.data.stats import FeatureStats
from hal.training.types import ModelInputs
from hal.training.zoo.preprocess.registry import InputPreprocessRegistry


def _preprocess_numeric_features(
    sample: Dict[str, np.ndarray], player: str, opponent: str, stats: Dict[str, FeatureStats]
) -> np.ndarray:
    """Preprocess numeric features for both players."""
    numeric_features = (
        PLAYER_INPUT_FEATURES_TO_NORMALIZE + PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE + PLAYER_POSITION
    )
    numeric_inputs = []
    for feature in numeric_features:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE[feature]
        for p in [player, opponent]:
            feature_name = f"{p}_{feature}"
            numeric_inputs.append(preprocess_fn(sample[feature_name], stats[feature_name]))  # pylint: disable=E1102
    return np.stack(numeric_inputs, axis=-1)


def _preprocess_categorical_features(
    sample: Dict[str, np.ndarray], player: str, opponent: str, stats: Dict[str, FeatureStats]
) -> Dict[str, np.ndarray]:
    """Preprocess categorical features for both players."""
    processed_features = {}
    for feature in PLAYER_INPUT_FEATURES_TO_EMBED:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE[feature]
        for p, prefix in [(player, "ego"), (opponent, "opponent")]:
            feature_name = f"{p}_{feature}"
            processed_features[f"{prefix}_{feature}"] = preprocess_fn(  # pylint: disable=E1102
                sample[feature_name], stats[feature_name]
            )
    return processed_features


@InputPreprocessRegistry.register("inputs_v0")
def preprocess_inputs_v0(
    sample: Dict[str, np.ndarray], input_len: int, player: str, stats: Dict[str, FeatureStats]
) -> ModelInputs:
    """Slice input sample to the input length."""
    assert player in VALID_PLAYERS
    opponent = "p2" if player == "p1" else "p1"

    input_sample = {k: v[:input_len] for k, v in sample.items()}

    stage = input_sample["stage"]
    categorical_features = _preprocess_categorical_features(input_sample, player, opponent, stats)
    gamestate = _preprocess_numeric_features(input_sample, player, opponent, stats)

    return ModelInputs(stage=stage, gamestate=gamestate, **categorical_features)
