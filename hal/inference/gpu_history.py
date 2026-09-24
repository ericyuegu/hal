"""GPU-resident mirrored context for one live policy stream."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from hal.training.context_history import ContextHistory
from hal.training.controller_codec import CONTROLLER_GROUP_COUNT
from hal.training.controller_codec import DiscreteControllerCodec
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import Context


class GpuContextHistory:
    """Mirror each row at aligned phases for zero-copy compiled feature views."""

    def __init__(self, history: ContextHistory, codec: DiscreteControllerCodec, device: torch.device) -> None:
        layout = history.layout
        length = history.L
        self.history = history
        self.length = length
        self.device = device
        self.codec = codec
        self._n_value = len(layout.value_names)
        self._float_row = np.empty(self._n_value + len(layout.mask_names), dtype=np.float32)
        self._cat_row = np.empty(len(layout.cat_names), dtype=np.int64)
        self._float_row[: self._n_value] = layout.zero_value
        self._float_row[self._n_value :] = layout.zero_mask
        self._cat_row[:] = layout.zero_cat
        float_zero = np.broadcast_to(self._float_row[None, :, None], (4, len(self._float_row), 2 * length + 4))
        cat_zero = np.broadcast_to(self._cat_row[None, :, None], (2, len(self._cat_row), 2 * length + 2))
        self.floats = torch.from_numpy(float_zero.copy()).to(device)
        self.cats = torch.from_numpy(cat_zero.copy()).to(device)
        neutral = codec.quantize(torch.as_tensor(NEUTRAL_ACTION, device=device).reshape(1, -1))[0]
        self.actions = neutral.expand(2 * length, CONTROLLER_GROUP_COUNT).clone()
        self._pad = torch.tensor([length], dtype=torch.long, device=device)
        self._player_id = torch.tensor([[0]], dtype=torch.long, device=device)

    def push(self, action: np.ndarray) -> None:
        """Mirror only the newest preprocessed row after ``history.push``."""
        history = self.history
        at = (history.written - 1) % self.length
        self._float_row[: self._n_value] = history.values[:, at]
        self._float_row[self._n_value :] = history.masks[:, at]
        self._cat_row[:] = history.cats[:, at]
        floats = torch.from_numpy(self._float_row)
        cats = torch.from_numpy(self._cat_row)
        for phase in range(4):
            self.floats[phase, :, at + phase].copy_(floats)
            self.floats[phase, :, at + self.length + phase].copy_(floats)
        for phase in range(2):
            self.cats[phase, :, at + phase].copy_(cats)
            self.cats[phase, :, at + self.length + phase].copy_(cats)
        indices = self.codec.quantize(torch.from_numpy(action).to(self.device).reshape(1, -1))[0]
        self.actions[at].copy_(indices)
        self.actions[at + self.length].copy_(indices)
        self._pad[0] = self.length - history.count

    def context(self, player_id: int, stream_id: int, reset: bool) -> Context:
        layout = self.history.layout
        start = self.history.written % self.length
        if self.history.count == self.length and layout.dpos_mask_row >= 0:
            for phase in range(4):
                for at in (start + phase, start + self.length + phase):
                    self.floats[phase, layout.dpos_rows, at] = 0.0
                    self.floats[phase, layout.dpos_mask_row, at] = 1.0
        self._player_id[0, 0] = player_id
        float_phase = -start % 4
        cat_phase = -start % 2
        window = self.floats[float_phase, :, start + float_phase : start + float_phase + self.length]
        cats = self.cats[cat_phase, :, start + cat_phase : start + cat_phase + self.length]
        features = {
            name: window[index].unsqueeze(0) for index, name in enumerate((*layout.value_names, *layout.mask_names))
        }
        features.update({name: cats[index].unsqueeze(0) for index, name in enumerate(layout.cat_names)})
        features["ego_player_id"] = self._player_id.expand(1, self.length)
        return Context(
            features={name: features[name] for name in sorted(features)},
            ctx_pad=self._pad,
            slot_ids=torch.tensor([stream_id], dtype=torch.long, device=self.device),
            reset=torch.tensor([reset], dtype=torch.bool, device=self.device),
        )

    def action_indices(self) -> Tensor:
        start = self.history.written % self.length
        return self.actions[start : start + self.length].unsqueeze(0)
