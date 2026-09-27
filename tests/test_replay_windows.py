"""Current replay-window operations extracted from the generic loader."""

import numpy as np

from hal.representation.features import FeatureProjection
from hal.training.replay_windows import choose_chunk_starts
from hal.training.replay_windows import collate_windows
from hal.training.replay_windows import make_window


def test_window_relabels_ego_before_projection_and_pads_cold_start() -> None:
    sample = {
        "frame": np.arange(6, dtype=np.int32),
        "p1_value": np.arange(6, dtype=np.float32),
        "p2_value": np.arange(6, dtype=np.float32) + 100,
        "schema_version": 7,
    }
    projection = FeatureProjection(frozenset({"frame", "ego_value"}))

    window = make_window(sample, ego_prefix="p2", start=-2, pad=2, length=5, projection=projection)

    assert set(window) == {"frame", "ego_value"}
    np.testing.assert_array_equal(window["frame"], np.array([0, 0, 0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(window["ego_value"], np.array([0, 0, 100, 101, 102], dtype=np.float32))
    stacked = collate_windows([window, window])
    assert stacked["ego_value"].shape == (2, 5)


def test_chunk_starts_match_control_rng_and_full_context_floor() -> None:
    standard = choose_chunk_starts(30, 6, 4, 3, np.random.default_rng(7))
    full_context = choose_chunk_starts(30, 6, 4, 3, np.random.default_rng(7), require_full_context=True)

    assert standard.tolist() == [4, 22]
    assert full_context.tolist() == [6, 26]
    assert choose_chunk_starts(9, 6, 4, 1, np.random.default_rng(7), require_full_context=True).size == 0
