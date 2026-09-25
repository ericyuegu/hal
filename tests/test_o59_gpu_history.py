"""The GPU-facing mirror exposes the exact chronological O59 context."""

import numpy as np
import torch

from hal.data.feature_stats import FeatureStats
from hal.inference.backends.history_decoder.gpu_history import GpuContextHistory
from hal.inference.backends.history_decoder.policy import _MODEL_FIELDS
from hal.training.context_history import ContextHistory
from hal.training.context_history import stack_context_windows
from hal.training.controller_codec import DiscreteControllerCodec
from hal.training.ego_stats import consolidate_key
from hal.training.features import ACTION_CHANNELS
from hal.training.features import BASE_ITEMS_PROJECTION
from hal.training.features import ITEM_COLUMNS
from hal.training.features import feature_kind
from hal.training.features import stack_actions


def _input() -> tuple[dict[str, float | int], dict[str, FeatureStats]]:
    flat = {name: 0 if feature_kind(name, ITEM_COLUMNS) in ("cat", "button") else 0.0 for name in _MODEL_FIELDS}
    stats = {
        consolidate_key(name): FeatureStats(1.0, 2.0, -10.0, 10.0)
        for name in _MODEL_FIELDS
        if feature_kind(name, ITEM_COLUMNS) not in ("cat", "button", "stick_trigger")
    }
    return flat, stats


def test_mirrored_context_matches_cpu_window_through_wrap() -> None:
    flat, stats = _input()
    codec = DiscreteControllerCodec(32)
    history = ContextHistory.from_frame(flat, "p1", stats, 8, ITEM_COLUMNS, BASE_ITEMS_PROJECTION)
    gpu = GpuContextHistory(history, codec, torch.device("cpu"))

    for frame in range(19):
        flat["ego_position_x"] = float(frame)
        action = np.zeros(len(ACTION_CHANNELS), dtype=np.float32)
        action[0] = (frame % 5) / 4
        history.gather(flat, action)
        history.push(None)
        gpu.push(action)
        actual = gpu.context(17, 4, frame == 0)
        expected = stack_context_windows((history,), 8).features("cpu", all_masks=True)
        expected["ego_player_id"] = torch.full((1, 8), 17, dtype=torch.long)
        assert actual.features.keys() == {name for name in expected}
        for name, value in expected.items():
            torch.testing.assert_close(actual.features[name], value, rtol=0, atol=0)
        torch.testing.assert_close(actual.ctx_pad, torch.tensor([8 - history.count]))
        action_indices = codec.quantize(stack_actions(expected))
        torch.testing.assert_close(gpu.action_indices(), action_indices)
        for phase in range(4):
            torch.testing.assert_close(
                gpu.floats[phase, :, phase : phase + 8], gpu.floats[phase, :, phase + 8 : phase + 16]
            )
        for phase in range(2):
            torch.testing.assert_close(
                gpu.cats[phase, :, phase : phase + 8], gpu.cats[phase, :, phase + 8 : phase + 16]
            )
