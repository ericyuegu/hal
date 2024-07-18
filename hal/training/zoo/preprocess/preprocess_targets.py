from typing import Dict

import numpy as np
from data.constants import STICK_XY_CLUSTER_CENTERS_V0

from hal.data.constants import VALID_PLAYERS
from hal.data.normalize import union
from hal.training.zoo.preprocess.encoding import get_closest_stick_xy_cluster_v0
from hal.training.zoo.preprocess.encoding import one_hot_2d
from hal.training.zoo.preprocess.encoding import one_hot_from_int
from hal.training.zoo.preprocess.registry import Player
from hal.training.zoo.preprocess.registry import TargetPreprocessRegistry


@TargetPreprocessRegistry.register("targets_v0")
def preprocess_targets_v0(sample: Dict[str, np.ndarray], player: Player) -> Dict[str, np.ndarray]:
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
    button_a = sample[f"{player}_button_a"]
    button_b = sample[f"{player}_button_b"]
    jump = union(sample[f"{player}_button_x"], sample[f"{player}_button_y"])
    button_z = sample[f"{player}_button_z"]
    shoulder = union(sample[f"{player}_button_l"], sample[f"{player}_button_r"])
    no_button = np.zeros_like(button_a)
    stacked_buttons = np.stack((button_a, button_b, jump, button_z, shoulder, no_button), axis=1)
    one_hot_buttons = one_hot_2d(stacked_buttons)

    return {
        "main_stick": one_hot_main_stick.astype(np.float32),
        "c_stick": one_hot_c_stick.astype(np.float32),
        "buttons": one_hot_buttons.astype(np.float32),
    }
