import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hal.training.zoo.preprocess.encoding import one_hot_2d


@pytest.mark.parametrize(
    "input_array, expected_output",
    [
        # Test case 0
        (
            np.array(
                [
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                ],
                dtype=np.int8,
            ),
        ),
        # Test case 1
        (
            np.array(
                [
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            np.array(
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
                ],
                dtype=np.int8,
            ),
        ),
        # Test case 2
        (
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
        ),
        # Test case 3
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
            np.array(
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                ],
                dtype=np.int8,
            ),
        ),
        # Test case 4
        (
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0],
                ]
            ),
        ),
    ],
)
def test_convert_target_to_one_hot_3d(input_array, expected_output) -> None:
    result = one_hot_2d(input_array)
    assert_array_equal(result, expected_output, err_msg=f"{result}\n{expected_output}")
