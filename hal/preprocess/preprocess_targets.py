from tensordict import TensorDict

from hal.constants import Player
from hal.constants import VALID_PLAYERS
from hal.preprocess.registry import TargetPreprocessRegistry
from hal.preprocess.transform import preprocess_controller_inputs_coarse
from hal.preprocess.transform import preprocess_controller_inputs_fine_shoulder


@TargetPreprocessRegistry.register("targets_v0")
def preprocess_targets_v0(sample: TensorDict, player: Player) -> TensorDict:
    """
    One-hot encode buttons and discretize main and c-stick x, y values for a given player.
    """
    assert player in VALID_PLAYERS
    controller_features = preprocess_controller_inputs_coarse(sample, player)
    batch_size = controller_features["main_stick"].shape[0]
    return TensorDict(controller_features, batch_size=(batch_size,))


@TargetPreprocessRegistry.register("targets_v1")
def preprocess_targets_v1(sample: TensorDict, player: Player) -> TensorDict:
    """
    One-hot encode buttons and discretize main, c-stick x, y values and analog shoulder presses for a given player.
    """
    assert player in VALID_PLAYERS
    controller_features = preprocess_controller_inputs_fine_shoulder(sample, player)
    batch_size = controller_features["main_stick"].shape[0]
    return TensorDict(controller_features, batch_size=(batch_size,))
