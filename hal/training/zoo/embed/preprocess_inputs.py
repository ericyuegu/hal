from typing import Dict

import numpy as np

from hal.data.preprocessing import NORMALIZATION_FN_BY_FEATURE
from hal.data.preprocessing import NormalizationFn
from hal.data.preprocessing import PLAYER_INPUT_FEATURES_TO_EMBED
from hal.data.preprocessing import PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE
from hal.data.preprocessing import PLAYER_INPUT_FEATURES_TO_NORMALIZE
from hal.data.preprocessing import PLAYER_POSITION
from hal.data.preprocessing import VALID_PLAYERS
from hal.data.stats import FeatureStats
from hal.training.zoo.embed.registry import Embed


def _preprocess_numeric_features(
    sample: Dict[str, np.ndarray], player: str, other_player: str, stats: Dict[str, FeatureStats]
) -> np.ndarray:
    """Preprocess numeric features for both players."""
    numeric_features = (
        PLAYER_INPUT_FEATURES_TO_NORMALIZE + PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE + PLAYER_POSITION
    )
    numeric_inputs = []
    for feature in numeric_features:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE[feature]
        for p in [player, other_player]:
            feature_name = f"{p}_{feature}"
            numeric_inputs.append(preprocess_fn(sample[feature_name], stats[feature_name]))  # pylint: disable=E1102
    return np.stack(numeric_inputs, axis=-1)


def _preprocess_categorical_features(
    sample: Dict[str, np.ndarray], player: str, other_player: str, stats: Dict[str, FeatureStats]
) -> Dict[str, np.ndarray]:
    """Preprocess categorical features for both players."""
    processed_features = {}
    for feature in PLAYER_INPUT_FEATURES_TO_EMBED:
        preprocess_fn: NormalizationFn = NORMALIZATION_FN_BY_FEATURE[feature]
        for p, suffix in [(player, "ego"), (other_player, "other")]:
            feature_name = f"{p}_{feature}"
            processed_features[f"{feature}_{suffix}"] = preprocess_fn(  # pylint: disable=E1102
                sample[feature_name], stats[feature_name]
            )
    return processed_features


def preprocess_inputs_v0(
    sample: Dict[str, np.ndarray], player: str, stats: Dict[str, FeatureStats]
) -> Dict[str, np.ndarray]:
    """Preprocess basic player state."""
    assert player in VALID_PLAYERS
    other_player = "p2" if player == "p1" else "p1"
    inputs = {"stage": sample["stage"]}

    inputs.update(_preprocess_categorical_features(sample, player, other_player, stats))
    inputs["gamestate"] = _preprocess_numeric_features(sample, player, other_player, stats)

    return inputs


Embed.register("inputs_v0", preprocess_inputs_v0)
