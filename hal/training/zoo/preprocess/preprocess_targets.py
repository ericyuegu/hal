from typing import Dict

import numpy as np

from hal.data.normalize import VALID_PLAYERS
from hal.data.normalize import union
from hal.data.stats import FeatureStats
from hal.training.types import ModelOutputs
from hal.training.zoo.preprocess.encoding import get_closest_stick_xy_cluster_v0
from hal.training.zoo.preprocess.encoding import one_hot_3d_fast_bugged
from hal.training.zoo.preprocess.registry import TargetPreprocessRegistry


@TargetPreprocessRegistry.register("targets_v0")
def preprocess_targets_v0(
    sample: Dict[str, np.ndarray], input_len: int, player: str, stats: Dict[str, FeatureStats]
) -> ModelOutputs:
    """
    Return only target features after the input trajectory length.

    One-hot encode buttons and discretize analog stick x, y values for a given player.
    """
    assert player in VALID_PLAYERS

    target_sample = {k: v[input_len:] for k, v in sample.items()}

    # Main stick and c-stick classification
    main_stick_x = target_sample[f"{player}_main_stick_x"]
    main_stick_y = target_sample[f"{player}_main_stick_y"]
    c_stick_x = target_sample[f"{player}_c_stick_x"]
    c_stick_y = target_sample[f"{player}_c_stick_y"]
    main_stick_clusters = get_closest_stick_xy_cluster_v0(main_stick_x, main_stick_y)
    c_stick_clusters = get_closest_stick_xy_cluster_v0(c_stick_x, c_stick_y)

    # Stack buttons and encode one_hot
    button_a = target_sample[f"{player}_button_a"]
    button_b = target_sample[f"{player}_button_b"]
    jump = union(target_sample[f"{player}_button_x"], target_sample[f"{player}_button_y"])
    button_z = target_sample[f"{player}_button_z"]
    shoulder = union(target_sample[f"{player}_button_l"], target_sample[f"{player}_button_r"])
    stacked_buttons = np.stack((button_a, button_b, jump, button_z, shoulder), axis=1)[np.newaxis, ...]
    buttons = one_hot_3d_fast_bugged(stacked_buttons)

    return ModelOutputs(main_stick=main_stick_clusters, c_stick=c_stick_clusters, buttons=buttons)
