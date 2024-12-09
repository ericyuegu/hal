import torch
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import STICK_XY_CLUSTER_CENTERS_V0
from hal.constants import VALID_PLAYERS
from hal.training.preprocess.registry import TargetPreprocessRegistry
from hal.training.preprocess.transform import encode_buttons_one_hot
from hal.training.preprocess.transform import get_closest_stick_xy_cluster_v0
from hal.training.preprocess.transform import one_hot_from_int


@TargetPreprocessRegistry.register("targets_v0")
def preprocess_targets_v0(sample: TensorDict, player: Player) -> TensorDict:
    """
    Return only target features after the input trajectory length.

    One-hot encode buttons and discretize analog stick x, y values for a given player.
    """
    assert player in VALID_PLAYERS

    # Main stick and c-stick classification
    main_stick_x = sample[f"{player}_main_stick_x"]
    main_stick_y = sample[f"{player}_main_stick_y"]
    c_stick_x = sample[f"{player}_c_stick_x"]
    c_stick_y = sample[f"{player}_c_stick_y"]
    main_stick_clusters = get_closest_stick_xy_cluster_v0(main_stick_x, main_stick_y)
    one_hot_main_stick = one_hot_from_int(main_stick_clusters, len(STICK_XY_CLUSTER_CENTERS_V0))
    c_stick_clusters = get_closest_stick_xy_cluster_v0(c_stick_x, c_stick_y)
    one_hot_c_stick = one_hot_from_int(c_stick_clusters, len(STICK_XY_CLUSTER_CENTERS_V0))

    # Stack buttons and encode one_hot
    button_a = sample[f"{player}_button_a"].bool()
    button_b = sample[f"{player}_button_b"].bool()
    button_x = sample[f"{player}_button_x"].bool()
    button_y = sample[f"{player}_button_y"].bool()
    button_z = sample[f"{player}_button_z"].bool()
    button_l = sample[f"{player}_button_l"].bool()
    button_r = sample[f"{player}_button_r"].bool()

    jump = button_x | button_y
    shoulder = button_l | button_r
    no_button = ~(button_a | button_b | jump | button_z | shoulder)

    stacked_buttons = torch.stack((button_a, button_b, jump, button_z, shoulder, no_button), dim=-1)
    one_hot_buttons = encode_buttons_one_hot(stacked_buttons.numpy())

    return TensorDict(
        {
            "main_stick": torch.tensor(one_hot_main_stick, dtype=torch.float32),
            "c_stick": torch.tensor(one_hot_c_stick, dtype=torch.float32),
            "buttons": torch.tensor(one_hot_buttons, dtype=torch.float32),
        },
        batch_size=(one_hot_main_stick.shape[0]),
    )


TARGETS_EMBEDDING_SIZES = {
    "targets_v0": {
        "main_stick": len(STICK_XY_CLUSTER_CENTERS_V0),
        "c_stick": len(STICK_XY_CLUSTER_CENTERS_V0),
        "buttons": 6,  # Number of button categories (a, b, jump, z, shoulder, no_button)
    }
}
