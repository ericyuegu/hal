from typing import Dict

import numpy as np

from hal.data.preprocessing import VALID_PLAYERS
from hal.data.preprocessing import get_closest_stick_xy_cluster_v0
from hal.data.preprocessing import one_hot_3d_fast_bugged
from hal.data.preprocessing import union
from hal.zoo.embed.registry import Embed


def preprocess_targets_v0(sample: Dict[str, np.ndarray], player: str) -> Dict[str, np.ndarray]:
    """One-hot encode buttons and discretize analog stick x, y values for a given player."""
    assert player in VALID_PLAYERS
    target = {}

    # Main stick and c-stick classification
    main_stick_x = sample[f"{player}_main_stick_x"]
    main_stick_y = sample[f"{player}_main_stick_y"]
    c_stick_x = sample[f"{player}_c_stick_x"]
    c_stick_y = sample[f"{player}_c_stick_y"]
    main_stick_clusters = get_closest_stick_xy_cluster_v0(main_stick_x, main_stick_y)
    c_stick_clusters = get_closest_stick_xy_cluster_v0(c_stick_x, c_stick_y)
    target["main_stick"] = main_stick_clusters
    target["c_stick"] = c_stick_clusters

    # Stack buttons and encode one_hot
    button_a = sample[f"{player}_button_a"]
    button_b = sample[f"{player}_button_b"]
    jump = union(sample[f"{player}_button_x"], sample[f"{player}_button_y"])
    button_z = sample[f"{player}_button_z"]
    shoulder = union(sample[f"{player}_button_l"], sample[f"{player}_button_r"])
    stacked_buttons = np.stack((button_a, button_b, jump, button_z, shoulder), axis=1)[np.newaxis, ...]
    target["buttons"] = one_hot_3d_fast_bugged(stacked_buttons)

    return target


Embed.register("targets_v0", preprocess_targets_v0)
