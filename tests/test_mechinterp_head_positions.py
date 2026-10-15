import numpy as np
import pytest

from research.mechinterp_20260911 import attention_head_positions


def test_distinct_head_maxima_preserve_key_positions_and_weights() -> None:
    weights = np.array([[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6]]], dtype=np.float32)
    (layer,) = attention_head_positions.summarize_positions(weights)
    assert layer.key_positions == (0, 1, 2)
    assert layer.peak_weights == tuple(float(value) for value in weights[0, range(3), range(3)])
    assert layer.distinct_positions == 3
    assert layer.head_pairs_same_position == 0


def test_shared_positions_count_unordered_head_pairs_within_each_layer() -> None:
    weights = np.array(
        [[[1.0, 0.0], [0.8, 0.2], [0.7, 0.3]], [[0.1, 0.9], [0.8, 0.2], [0.9, 0.1]]],
        dtype=np.float32,
    )
    first, second = attention_head_positions.summarize_positions(weights)
    assert first.key_positions == (0, 0, 0)
    assert first.distinct_positions == 1
    assert first.head_pairs_same_position == 3
    assert second.key_positions == (1, 0, 0)
    assert second.distinct_positions == 2
    assert second.head_pairs_same_position == 1


def test_tied_maxima_are_rejected_instead_of_arbitrarily_choosing_a_position() -> None:
    with pytest.raises(ValueError, match="tied positions"):
        attention_head_positions.summarize_positions(np.array([[[0.5, 0.5]]], dtype=np.float32))


@pytest.mark.parametrize("weights", [[[[float("nan"), 0.0]]], [[[1.1, -0.1]]]])
def test_invalid_probabilities_are_rejected(weights: list[list[list[float]]]) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        attention_head_positions.summarize_positions(np.asarray(weights, dtype=np.float32))


def test_unnormalized_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        attention_head_positions.summarize_positions(np.array([[[0.6, 0.1]]], dtype=np.float32))


def test_wrong_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty float32"):
        attention_head_positions.summarize_positions(np.array([[0.8, 0.2]], dtype=np.float32))
