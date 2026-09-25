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


class GpuTokenHistory:
    """Stage only the last two preprocessed frames for incremental inference."""

    def __init__(self, history: ContextHistory, codec: DiscreteControllerCodec, device: torch.device) -> None:
        if history.layout.spatial_at is not None:
            raise ValueError("KV cache token staging requires per-frame features without window-relative spatial data")
        self.history = history
        self.codec = codec
        self.device = device
        layout = history.layout
        self._float_host = np.zeros((len(layout.value_names) + len(layout.mask_names), 2), dtype=np.float32)
        self._cat_host = np.zeros((len(layout.cat_names), 2), dtype=np.int64)
        self._action_host = np.zeros((2, len(NEUTRAL_ACTION)), dtype=np.float32)
        self.floats = torch.zeros_like(torch.from_numpy(self._float_host), device=device)
        self.cats = torch.zeros_like(torch.from_numpy(self._cat_host), device=device)
        self.actions = codec.quantize(torch.from_numpy(self._action_host).to(device)).unsqueeze(0)
        self.player = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._dirty = False

    def push(self, action: np.ndarray) -> None:
        at = (self.history.written - 1) % self.history.L
        values = len(self.history.layout.value_names)
        self._float_host[:, 0] = self._float_host[:, 1]
        self._cat_host[:, 0] = self._cat_host[:, 1]
        self._action_host[0] = self._action_host[1]
        self._float_host[:values, 1] = self.history.values[:, at]
        self._float_host[values:, 1] = self.history.masks[:, at]
        self._cat_host[:, 1] = self.history.cats[:, at]
        self._action_host[1] = action
        self._dirty = True

    def context(self, player_id: int, stream_id: int, reset: bool) -> Context:
        if self._dirty:
            self.floats.copy_(torch.from_numpy(self._float_host))
            self.cats.copy_(torch.from_numpy(self._cat_host))
            self.actions = self.codec.quantize(torch.from_numpy(self._action_host).to(self.device)).unsqueeze(0)
            self._dirty = False
        self.player.fill_(player_id)
        features = self.features_from(self.floats, self.cats, self.player)
        # RNG identity metadata belongs on the host; it is read by Python each plan.
        return Context(
            features,
            torch.tensor([max(0, 2 - self.history.count)], device=self.device),
            torch.tensor([stream_id]),
            torch.tensor([reset]),
        )

    def features_from(self, floats: Tensor, cats: Tensor, player: Tensor) -> dict[str, Tensor]:
        layout = self.history.layout
        features = {
            name: floats[index : index + 1] for index, name in enumerate((*layout.value_names, *layout.mask_names))
        }
        features.update({name: cats[index : index + 1] for index, name in enumerate(layout.cat_names)})
        features["ego_player_id"] = player.expand(1, 2)
        return features

    def action_indices(self) -> Tensor:
        if self._dirty:
            raise RuntimeError("read token context before its action indices")
        return self.actions
