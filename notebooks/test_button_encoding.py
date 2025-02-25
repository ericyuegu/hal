# %%

import numpy as np

from hal.preprocess.transformations import convert_multi_hot_to_one_hot_early_release

buttons_LD = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
)

convert_multi_hot_to_one_hot_early_release(buttons_LD)
