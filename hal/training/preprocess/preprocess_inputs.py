from functools import partial
from typing import Dict

from tensordict import TensorDict

from hal.constants import PLAYER_INPUT_FEATURES_TO_EMBED
from hal.constants import PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE
from hal.constants import PLAYER_INPUT_FEATURES_TO_NORMALIZE
from hal.constants import PLAYER_POSITION
from hal.constants import Player
from hal.constants import STAGE
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import TARGET_FEATURES_TO_ONE_HOT_ENCODE
from hal.data.normalize import NormalizationFn
from hal.data.normalize import cast_int32
from hal.data.normalize import invert_and_normalize
from hal.data.normalize import normalize
from hal.data.normalize import normalize_and_embed_fourier
from hal.data.normalize import standardize
from hal.data.stats import FeatureStats
from hal.training.config import DataConfig
from hal.training.preprocess.base import preprocess_input_features
from hal.training.preprocess.config import InputPreprocessConfig
from hal.training.preprocess.preprocess_targets import preprocess_targets_v0
from hal.training.preprocess.registry import InputPreprocessRegistry


@InputPreprocessRegistry.register("inputs_v0", InputPreprocessConfig.v0())
def preprocess_inputs_v0(
    sample: TensorDict, data_config: DataConfig, ego: Player, stats: Dict[str, FeatureStats]
) -> TensorDict:
    """Slice input sample to the input length.

    Expects tensordict with shape (trajectory_len,)
    """
    trajectory_len = data_config.input_len + data_config.target_len

    return TensorDict(
        preprocess_input_features(
            sample=sample[:trajectory_len],
            ego=ego,
            config=InputPreprocessConfig.v0(),
            stats=stats,
        ),
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
    normalization_fn_by_feature_name: Dict[str, NormalizationFn] = {
        **dict.fromkeys(STAGE, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_EMBED, cast_int32),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_NORMALIZE, normalize),
        **dict.fromkeys(PLAYER_INPUT_FEATURES_TO_INVERT_AND_NORMALIZE, invert_and_normalize),
        **dict.fromkeys(PLAYER_POSITION, partial(normalize_and_embed_fourier, dim=8)),
    }

    return _preprocess_features_by_mapping(
        sample=sample,
        ego=ego,
        stats=stats,
        player_numeric_feature_names=numeric_features,
        normalization_fn_by_feature_name=normalization_fn_by_feature_name,
        batch_size=(trajectory_len,),
    )


@InputPreprocessRegistry.register(
    "inputs_v2",
    num_features=2 * (len(PLAYER_NUMERIC_FEATURES_V0) + len(STICK_XY_CLUSTER_CENTERS_V0))
    + len(TARGET_FEATURES_TO_ONE_HOT_ENCODE),
)
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
