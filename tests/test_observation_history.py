"""Incremental live observations reproduce offline 059 feature windows."""

import numpy as np
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.inference.observation_history import ObservationHistory
from hal.inference.observation_history import stack_observation_windows
from hal.representation.features import ACTION_CHANNELS
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind
from hal.representation.features import preprocess
from hal.wire import ACTION_DIM


def _frame(frame: int) -> dict[str, float | int]:
    columns = BASE_ITEMS_PROJECTION.columns - {f"ego_{name}" for name in ACTION_CHANNELS}
    out: dict[str, float | int] = {}
    for name in columns:
        kind = feature_kind(name, ITEM_COLUMNS)
        if kind == "float":
            out[name] = np.float32((frame % 17) * 0.25) if "position" in name else np.float32(frame % 9)
        else:
            out[name] = frame % 3 if "character" in name else 0
    if frame % 19 == 0:
        out["ego_position_x"] = float("nan")
    return out


def _stats() -> dict[str, FeatureStats]:
    columns = BASE_ITEMS_PROJECTION.columns - {f"ego_{name}" for name in ACTION_CHANNELS}
    return {
        consolidate_key(name): FeatureStats(mean=1.0, std=2.0, min=-10.0, max=20.0)
        for name in columns
        if feature_kind(name, ITEM_COLUMNS) == "float"
    }


def _reference(
    rows: list[tuple[dict[str, float | int], np.ndarray]], length: int, stats: dict[str, FeatureStats]
) -> dict[str, torch.Tensor]:
    pad = length - len(rows)
    columns: dict[str, np.ndarray] = {}
    for name, value in rows[0][0].items():
        dtype = np.int32 if isinstance(value, int) else np.float32
        values = [0] * pad + [frame[name] for frame, _ in rows]
        columns[name] = np.asarray(values, dtype=dtype)[None, :]
    actions = np.stack([np.zeros(ACTION_DIM, dtype=np.float32)] * pad + [action for _, action in rows])
    for index, channel in enumerate(ACTION_CHANNELS):
        values = actions[:, index]
        columns[f"ego_{channel}"] = ((values > 0.5).astype(np.int32) if channel.startswith("button_") else values)[
            None, :
        ]
    return preprocess(columns, stats, extra=ITEM_COLUMNS, projection=BASE_ITEMS_PROJECTION)


def test_mirrored_observation_ring_matches_offline_windows_through_reset_and_wraps() -> None:
    length = 32
    stats = _stats()
    history: ObservationHistory | None = None
    rows: list[tuple[dict[str, float | int], np.ndarray]] = []
    previous_frame: int | None = None
    varied = 0
    for tick, frame_id in enumerate((*range(140, 210), *range(-123, -43))):
        if previous_frame is not None and frame_id < previous_frame:
            history = None
            rows.clear()
        frame = _frame(tick)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[:6] = ((tick % 11) - 5) / 10.0
        action[6:] = float(tick % 2)
        if history is None:
            history = ObservationHistory.from_frame(frame, "p1", stats, length, ITEM_COLUMNS, BASE_ITEMS_PROJECTION)
        history.gather(frame, action)
        history.push()
        rows.append((frame, action))
        rows = rows[-length:]
        actual = stack_observation_windows((history,), length).features("cpu")
        expected = _reference(rows, length, stats)
        assert actual.keys() == expected.keys()
        for name in expected:
            assert torch.equal(actual[name], expected[name]), f"frame {frame_id} feature {name} differs"
            varied += int(bool(torch.any(expected[name] != expected[name].flatten()[0])))
        assert length - history.count == length - len(rows)
        previous_frame = frame_id
    assert varied > 1000
