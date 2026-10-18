"""Packed adapter features retain the Python column builder's exact arithmetic."""

import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.inference.column_context import ColumnContextBuilder
from hal.inference.column_context import PackedColumnContextBuilder
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind
from hal.sim.native import OBSERVATION_FIELDS
from hal.wire import ACTION_DIM
from hal.wire import MASK_INT32


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("constant", [False, True])
def test_packed_context_matches_column_builder_bitwise(device: str, constant: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    stats = {
        consolidate_key(name): FeatureStats(1.3, 0.0 if constant else 3.7, -2.1, -2.1 if constant else 5.9)
        for name in BASE_ITEMS_PROJECTION.columns
        if feature_kind(name, ITEM_COLUMNS) == "float"
    }
    rng = np.random.default_rng(65)
    packed = np.zeros((4, 13, len(OBSERVATION_FIELDS)), dtype=np.int32)
    columns = {}
    for index, name in enumerate(OBSERVATION_FIELDS):
        if np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i":
            values = rng.integers(0, 10, (4, 13), dtype=np.int32)
            values[:, 3] = MASK_INT32
        else:
            values = rng.normal(size=(4, 13)).astype(np.float32)
            values[:, 3] = np.nan
        packed[..., index] = values.view(np.int32)
        columns[name] = torch.tensor(values, device=device)
    records = torch.tensor(packed, device=device).view(torch.uint8)
    actions = torch.tensor(rng.random((4, 13, 2, ACTION_DIM), dtype=np.float32), device=device)
    sides = torch.tensor([1, 0, 0, 1], device=device)
    identities = torch.tensor([3, 2, 1, 4], device=device)
    lengths = torch.tensor([13, 5, 1, 11], device=device)
    builder = PackedColumnContextBuilder(stats, OBSERVATION_FIELDS, torch.device(device))
    actual = builder(records, actions, sides, identities, lengths)
    for row, side in enumerate((1, 0, 0, 1)):
        segment = torch.zeros(1, 13, device=device, dtype=torch.long)
        expected = ColumnContextBuilder(stats, side + 1)(
            {name: value[row : row + 1] for name, value in columns.items()},
            actions[row : row + 1],
            segment,
            torch.zeros_like(segment, dtype=torch.bool),
            identities[row : row + 1],
        )
        valid = torch.arange(13, device=device) < lengths[row]
        assert actual.keys() == expected.features.keys()
        for name, value in expected.features.items():
            torch.testing.assert_close(actual[name][row], torch.where(valid, value[0], 0), rtol=0, atol=0)
