"""The O59 action chunk must line up with Slippi's two-frame transport queue."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.o59 import O59Policy
from hal.inference.o59_model import CONTROLLER_GROUP_NAMES
from hal.inference.o59_model import Architecture
from hal.inference.transport import ActionTransport
from hal.wire import ACTION_CHANNELS


def test_o59_plan_sends_offsets_three_and_four_after_two_pending_frames() -> None:
    assert Architecture().head_offsets[:4] == (1, 2, 3, 4)
    groups = len(CONTROLLER_GROUP_NAMES)

    class Codec:
        def quantize(self, actions: torch.Tensor) -> torch.Tensor:
            assert actions.shape == (1, 2, len(ACTION_CHANNELS))
            forced = torch.zeros((1, 2, groups), dtype=torch.long)
            forced[0, :, 0] = (actions[0, :, 0] * 10).round().long()
            return forced

        def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
            values = torch.zeros((1, 4, len(ACTION_CHANNELS)))
            values[0, :, 0] = indices[0, :, 0].float() / 10
            return values

    policy = O59Policy.__new__(O59Policy)
    temporal = Mock()
    policy.model = SimpleNamespace(codec=Codec(), temporal=temporal, head_offsets=Architecture().head_offsets)
    policy.device = torch.device("cpu")
    policy._rng = Mock()
    policy._rng.uniforms.return_value = torch.zeros(1)
    policy._trunk = lambda *_args: torch.zeros(1, 1, 1)
    decoded = torch.arange(1, 5).reshape(1, 4, 1).expand(1, 4, groups)
    temporal.sample_indices.return_value = decoded
    policy._decoder = policy._sample
    policy.decode_seconds = []
    gpu = Mock()
    gpu.context.return_value = SimpleNamespace(features={}, ctx_pad=torch.zeros(1))
    gpu.action_indices.return_value = torch.zeros((1, 1, groups), dtype=torch.long)
    stream = SimpleNamespace(gpu=gpu, player_id=0, reset_pending=True, queued=deque())
    pending = (
        ControllerAction(0.7, 0, 0, 0, 0, 0, 0),
        ControllerAction(0.8, 0, 0, 0, 0, 0, 0),
    )
    item = PolicyInput(0, 0, 1, {}, NEUTRAL_CONTROLLER_ACTION, pending)

    policy._plan(item, stream)

    assert temporal.sample_indices.call_args.args[2] == (1, 2, 3, 4)
    forced = temporal.sample_indices.call_args.kwargs["forced_prefix"]
    assert forced[0, :, 0].tolist() == [7, 8]
    planned = tuple(stream.queued)
    assert [action.main_x for action in planned] == torch.tensor([0.3, 0.4]).tolist()
    transport = ActionTransport(2)
    due = (
        transport.submit(planned[0]),
        transport.submit(planned[1]),
        transport.submit(NEUTRAL_CONTROLLER_ACTION),
        transport.submit(NEUTRAL_CONTROLLER_ACTION),
    )
    assert due == (NEUTRAL_CONTROLLER_ACTION, NEUTRAL_CONTROLLER_ACTION, *planned)
