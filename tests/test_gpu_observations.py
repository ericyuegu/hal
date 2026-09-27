"""Cached GPU updates preserve the canonical CPU observation rows."""

import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.inference.gpu_observations import GpuObservationBatch
from hal.inference.gpu_observations import GpuObservationUpdates
from hal.inference.observation_history import ObservationHistory
from hal.inference.observation_history import stack_observation_windows
from hal.models.controller_codec import DiscreteControllerCodec
from hal.representation.features import ACTION_CHANNELS
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import stack_actions


def _history() -> tuple[ObservationHistory, dict[str, float | int]]:
    flat: dict[str, float | int] = {"p1_position_x": 0.0}
    stats = {consolidate_key("ego_position_x"): FeatureStats(1.0, 2.0, -10.0, 10.0)}
    return ObservationHistory.from_frame(flat, "p1", stats, 8, ITEM_COLUMNS, BASE_ITEMS_PROJECTION), flat


def _push(history: ObservationHistory, flat: dict[str, float | int], frame: int) -> np.ndarray:
    flat["p1_position_x"] = float(frame)
    action = np.zeros(len(ACTION_CHANNELS), dtype=np.float32)
    action[0] = (frame % 5) / 4
    history.gather(flat, action)
    history.push()
    return action


def test_ready_update_batch_preserves_stream_rows_and_dummy_inputs() -> None:
    codec = DiscreteControllerCodec(32)
    histories = [_history() for _ in range(3)]
    stages = [GpuObservationUpdates(history, codec, torch.device("cpu"), 4) for history, _ in histories]
    for stream, ((history, flat), stage) in enumerate(zip(histories, stages, strict=True)):
        for frame in range(stream + 1):
            stage.push(_push(history, flat, frame + stream * 10))
    batch = GpuObservationBatch(stages[0], 4)
    ready = (stages[2], stages[0])
    batch.gather(ready, (23, 17))
    features = batch.features()
    for row, (stage, player_id) in enumerate(zip(ready, (23, 17), strict=True)):
        expected = stage.context(player_id).features
        for name, value in expected.items():
            torch.testing.assert_close(features[name][row : row + 1], value, rtol=0, atol=0)
        torch.testing.assert_close(batch.actions[row : row + 1], stage.action_indices(), rtol=0, atol=0)
    assert bool((batch.player[2:] == 0).all())
    assert bool((batch.floats[2:] == 0).all())
    assert bool((batch.cats[2:] == 0).all())
    assert bool((batch.actions[2:] == 0).all())
    with pytest.raises(ValueError, match="fit"):
        batch.gather((), ())


@pytest.mark.parametrize("update_frames", [1, 2, 4])
def test_token_stage_matches_cpu_features_and_actions_through_wrap(update_frames: int) -> None:
    history, flat = _history()
    codec = DiscreteControllerCodec(32)
    updates = GpuObservationUpdates(history, codec, torch.device("cpu"), update_frames)
    for frame in range(30):
        action = _push(history, flat, frame)
        updates.push(action)
        staged = updates.context(17)
        expected = stack_observation_windows((history,), history.L).features("cpu", all_masks=True)
        expected["ego_player_id"] = torch.full((1, history.L), 17, dtype=torch.long)
        count = min(frame + 1, update_frames)
        for name, value in staged.features.items():
            torch.testing.assert_close(value[:, -count:], expected[name][:, -count:], rtol=0, atol=0)
        torch.testing.assert_close(
            updates.action_indices()[:, -count:], codec.quantize(stack_actions(expected))[:, -count:], rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_update_storage_is_stable_across_uploads() -> None:
    history, flat = _history()
    codec = DiscreteControllerCodec(32).cuda()
    stage = GpuObservationUpdates(history, codec, torch.device("cuda"), 4)
    addresses = tuple(tensor.data_ptr() for tensor in (stage.floats, stage.cats, stage.actions, stage.player))
    for frame in range(6):
        stage.push(_push(history, flat, frame))
        stage.upload(17)
        assert (
            tuple(tensor.data_ptr() for tensor in (stage.floats, stage.cats, stage.actions, stage.player)) == addresses
        )
