# %%
import numpy as np
from numpy.testing import assert_array_equal

from hal.data.preprocessing import convert_target_button_to_one_hot


def test_vectorized_custom_one_hot() -> None:
    arr = np.array(
        [
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        ],
        dtype=np.int8,
    )

    expected = np.array(
        [
            [
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
            ]
        ],
        dtype=np.int8,
    )

    print(convert_target_button_to_one_hot(arr) != expected)
    assert_array_equal(convert_target_button_to_one_hot(arr), expected)

    print("All test cases passed!")


# Run the tests
test_vectorized_custom_one_hot()
