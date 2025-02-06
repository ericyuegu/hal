from tensordict import TensorDict

from hal.constants import Player
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import VALID_PLAYERS
from hal.training.preprocess.registry import TargetPreprocessRegistry
from hal.training.preprocess.transform import preprocess_controller_inputs


@TargetPreprocessRegistry.register("targets_v0")
def preprocess_targets_v0(sample: TensorDict, player: Player) -> TensorDict:
    """
    Return only target features after the input trajectory length.

    One-hot encode buttons and discretize analog stick x, y values for a given player.
    """
    assert player in VALID_PLAYERS

    # Use the shared controller preprocessing helper
    controller_features = preprocess_controller_inputs(sample, player)
    batch_size = controller_features["main_stick"].shape[0]
    return TensorDict(controller_features, batch_size=(batch_size,))


TARGETS_EMBEDDING_SIZES = {
    "targets_v0": {
        "main_stick": len(STICK_XY_CLUSTER_CENTERS_V0),
        "c_stick": len(STICK_XY_CLUSTER_CENTERS_V0),
        "buttons": 6,  # Number of button categories (a, b, jump, z, shoulder, no_button)
    }
}
