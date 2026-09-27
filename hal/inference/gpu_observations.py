"""Prepared observation storage for dense windows and cached updates."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from hal.inference.observation_history import ObservationHistory
from hal.inference.observation_history import ObservationLayout
from hal.models.controller_codec import CONTROLLER_GROUP_COUNT
from hal.models.controller_codec import DiscreteControllerCodec
from hal.representation.features import NEUTRAL_ACTION
from hal.representation.features import Context


class GpuObservationWindow:
    """Mirror each row at aligned phases for zero-copy compiled feature views."""

    def __init__(self, history: ObservationHistory, codec: DiscreteControllerCodec, device: torch.device) -> None:
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
        self._neutral_indices = neutral.clone()
        self.actions = neutral.expand(2 * length, CONTROLLER_GROUP_COUNT).clone()
        self._pad = torch.tensor([length], dtype=torch.long, device=device)
        self._player_id = torch.tensor([[0]], dtype=torch.long, device=device)

    def reset(self) -> None:
        """Reuse the prepared window after its paired observation history resets."""
        layout = self.history.layout
        self._float_row[: self._n_value] = layout.zero_value
        self._float_row[self._n_value :] = layout.zero_mask
        self._cat_row[:] = layout.zero_cat
        self.floats.copy_(torch.from_numpy(self._float_row).view(1, -1, 1))
        self.cats.copy_(torch.from_numpy(self._cat_row).view(1, -1, 1))
        self.actions.copy_(self._neutral_indices.expand_as(self.actions))
        self._pad.fill_(self.length)
        self._player_id.zero_()

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

    def context(self, player_id: int) -> Context:
        layout = self.history.layout
        start = self.history.written % self.length
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
        )

    def action_indices(self) -> Tensor:
        start = self.history.written % self.length
        return self.actions[start : start + self.length].unsqueeze(0)


class GpuObservationWindowBatch:
    """Gather ready dense windows into one prepared GPU batch without a CPU copy."""

    def __init__(self, example: GpuObservationWindow, bucket: int) -> None:
        if bucket < 1 or bucket & (bucket - 1):
            raise ValueError("window batch bucket must be a positive power of two")
        self.bucket = bucket
        self.length = example.length
        self.device = example.device
        self.layout = example.history.layout
        self.floats = example.floats.new_empty((bucket, len(example._float_row), self.length))
        self.cats = example.cats.new_empty((bucket, len(example._cat_row), self.length))
        self.actions = example.actions.new_empty((bucket, self.length, CONTROLLER_GROUP_COUNT))
        self.player = example._player_id.new_empty((bucket, 1))
        self.ctx_pad = example._pad.new_empty(bucket)
        self.rows = 0

    def gather(self, windows: tuple[GpuObservationWindow, ...], player_ids: tuple[int, ...]) -> Context:
        if not windows or len(windows) != len(player_ids) or len(windows) > self.bucket:
            raise ValueError("ready dense windows do not fit their prepared bucket")
        for row, (window, player_id) in enumerate(zip(windows, player_ids, strict=True)):
            layout = window.history.layout
            if (
                window.device != self.device
                or window.length != self.length
                or (
                    layout.value_names != self.layout.value_names
                    or layout.mask_names != self.layout.mask_names
                    or layout.cat_names != self.layout.cat_names
                )
            ):
                raise ValueError("ready dense window layouts or devices do not match")
            if not isinstance(player_id, int) or isinstance(player_id, bool) or player_id < 0:
                raise ValueError("ready dense window player IDs must be non-negative integers")
            start = window.history.written % self.length
            float_phase = -start % 4
            cat_phase = -start % 2
            self.floats[row].copy_(
                window.floats[float_phase, :, start + float_phase : start + float_phase + self.length]
            )
            self.cats[row].copy_(window.cats[cat_phase, :, start + cat_phase : start + cat_phase + self.length])
            self.actions[row].copy_(window.actions[start : start + self.length])
            self.player[row, 0] = player_id
            self.ctx_pad[row] = self.length - window.history.count
        self.rows = len(windows)
        return Context(
            packed_features(
                self.layout,
                self.floats[: self.rows],
                self.cats[: self.rows],
                self.player[: self.rows],
            ),
            self.ctx_pad[: self.rows],
        )

    def action_indices(self) -> Tensor:
        if self.rows == 0:
            raise RuntimeError("dense windows were not gathered")
        return self.actions[: self.rows]


class GpuObservationUpdates:
    """Stage the last update's preprocessed frames for incremental inference."""

    def __init__(
        self, history: ObservationHistory, codec: DiscreteControllerCodec, device: torch.device, update_frames: int = 2
    ) -> None:
        if update_frames not in (1, 2, 4):
            raise ValueError("KV cache token staging supports one, two, or four frames")
        self.history = history
        self.codec = codec
        self.device = device
        self.update_frames = update_frames
        layout = history.layout
        self._float_host = np.zeros(
            (len(layout.value_names) + len(layout.mask_names), update_frames), dtype=np.float32
        )
        self._cat_host = np.zeros((len(layout.cat_names), update_frames), dtype=np.int64)
        self._action_host = np.zeros((update_frames, len(NEUTRAL_ACTION)), dtype=np.float32)
        self.floats = torch.zeros_like(torch.from_numpy(self._float_host), device=device)
        self.cats = torch.zeros_like(torch.from_numpy(self._cat_host), device=device)
        self.actions = codec.quantize(torch.from_numpy(self._action_host).to(device)).unsqueeze(0)
        self.player = torch.zeros((1, 1), dtype=torch.long, device=device)
        self._dirty = False

    def reset(self) -> None:
        """Clear one prepared stream's staged observations in place."""
        self._float_host.fill(0)
        self._cat_host.fill(0)
        self._action_host.fill(0)
        self.player.zero_()
        self._dirty = True

    def push(self, action: np.ndarray) -> None:
        at = (self.history.written - 1) % self.history.L
        values = len(self.history.layout.value_names)
        self._float_host[:, :-1] = self._float_host[:, 1:]
        self._cat_host[:, :-1] = self._cat_host[:, 1:]
        self._action_host[:-1] = self._action_host[1:]
        self._float_host[:values, -1] = self.history.values[:, at]
        self._float_host[values:, -1] = self.history.masks[:, at]
        self._cat_host[:, -1] = self.history.cats[:, at]
        self._action_host[-1] = action
        self._dirty = True

    def upload(self, player_id: int) -> None:
        """Copy pending host rows once before a graph replay or eager read."""
        if self._dirty:
            self.floats.copy_(torch.from_numpy(self._float_host))
            self.cats.copy_(torch.from_numpy(self._cat_host))
            self.actions.copy_(self.codec.quantize(torch.from_numpy(self._action_host).to(self.device)).unsqueeze(0))
            self._dirty = False
        self.player.fill_(player_id)

    def context(self, player_id: int) -> Context:
        self.upload(player_id)
        features = self.features_from(self.floats, self.cats, self.player)
        return Context(features, torch.tensor([max(0, self.update_frames - self.history.count)]))

    def features_from(self, floats: Tensor, cats: Tensor, player: Tensor) -> dict[str, Tensor]:
        layout = self.history.layout
        return packed_features(layout, floats, cats, player)

    def action_indices(self) -> Tensor:
        if self._dirty:
            raise RuntimeError("read token context before its action indices")
        return self.actions


def packed_features(layout: ObservationLayout, floats: Tensor, cats: Tensor, player: Tensor) -> dict[str, Tensor]:
    """Address the same named columns in one stream or a prepared batch."""
    if floats.ndim == 2:
        floats = floats.unsqueeze(0)
        cats = cats.unsqueeze(0)
    if floats.ndim != 3 or cats.ndim != 3 or player.shape != (floats.shape[0], 1):
        raise ValueError("packed observations must be [B, columns, frames] with player [B, 1]")
    if cats.shape[0] != floats.shape[0] or cats.shape[2] != floats.shape[2]:
        raise ValueError("float and categorical observation shapes do not agree")
    names = (*layout.value_names, *layout.mask_names)
    if floats.shape[1] != len(names) or cats.shape[1] != len(layout.cat_names):
        raise ValueError("packed observation column count does not match the layout")
    features = {name: floats[:, index] for index, name in enumerate(names)}
    features.update({name: cats[:, index] for index, name in enumerate(layout.cat_names)})
    features["ego_player_id"] = player.expand(floats.shape[0], floats.shape[2])
    return features


class GpuObservationBatch:
    """Static input storage for one prepared update shape and batch bucket."""

    def __init__(self, example: GpuObservationUpdates, bucket: int, frames: int | None = None) -> None:
        if bucket < 1 or bucket & (bucket - 1):
            raise ValueError("observation batch bucket must be a positive power of two")
        if frames is None:
            frames = example.update_frames
        if frames not in (1, 2, 4) or frames > example.update_frames:
            raise ValueError("prepared observation update must contain one, two, or four available frames")
        self.bucket = bucket
        self.update_frames = frames
        self.layout = example.history.layout
        self.floats = example.floats.new_zeros((bucket, example.floats.shape[0], frames))
        self.cats = example.cats.new_zeros((bucket, example.cats.shape[0], frames))
        self.actions = example.actions.new_zeros((bucket, frames, CONTROLLER_GROUP_COUNT))
        self.player = example.player.new_zeros((bucket, 1))

    def gather(
        self,
        updates: tuple[GpuObservationUpdates, ...],
        player_ids: tuple[int, ...],
        columns: slice | None = None,
    ) -> None:
        if not updates or len(updates) != len(player_ids) or len(updates) > self.bucket:
            raise ValueError("ready observation batch does not fit its prepared bucket")
        if columns is None:
            columns = slice(-self.update_frames, None)
        for row, (stage, player_id) in enumerate(zip(updates, player_ids, strict=True)):
            stage_layout = stage.history.layout
            if stage.update_frames < self.update_frames or (
                stage_layout.value_names != self.layout.value_names
                or stage_layout.mask_names != self.layout.mask_names
                or stage_layout.cat_names != self.layout.cat_names
            ):
                raise ValueError("ready observation layouts or update shapes do not match")
            stage.upload(player_id)
            if stage.floats[:, columns].shape[1] != self.update_frames:
                raise ValueError("selected observation columns do not match the prepared update shape")
            self.floats[row].copy_(stage.floats[:, columns])
            self.cats[row].copy_(stage.cats[:, columns])
            self.actions[row].copy_(stage.actions[0, columns])
            self.player[row].copy_(stage.player[0])
        if len(updates) < self.bucket:
            self.floats[len(updates) :].zero_()
            self.cats[len(updates) :].zero_()
            self.actions[len(updates) :].zero_()
            self.player[len(updates) :].zero_()

    def features(self) -> dict[str, Tensor]:
        return packed_features(self.layout, self.floats, self.cats, self.player)
