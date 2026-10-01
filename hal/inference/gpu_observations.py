"""Prepared observation storage for cached inference updates."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from hal.inference.observation_history import ObservationHistory
from hal.inference.observation_history import ObservationLayout
from hal.models.controller_codec import DiscreteControllerCodec
from hal.representation.features import NEUTRAL_ACTION
from hal.representation.features import Context


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
        self.codec = example.codec
        self._float_host = np.zeros((bucket, example.floats.shape[0], frames), dtype=np.float32)
        self._cat_host = np.zeros((bucket, example.cats.shape[0], frames), dtype=np.int64)
        self._action_host = np.zeros((bucket, frames, len(NEUTRAL_ACTION)), dtype=np.float32)
        self._player_host = np.zeros((bucket, 1), dtype=np.int64)
        self.floats = example.floats.new_zeros((bucket, example.floats.shape[0], frames))
        self.cats = example.cats.new_zeros((bucket, example.cats.shape[0], frames))
        self.raw_actions = torch.zeros_like(torch.from_numpy(self._action_host), device=example.device)
        self.player = example.player.new_zeros((bucket, 1))

    @property
    def actions(self) -> Tensor:
        return self.codec.quantize(self.raw_actions)

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
            if stage._float_host[:, columns].shape[1] != self.update_frames:
                raise ValueError("selected observation columns do not match the prepared update shape")
            self._float_host[row] = stage._float_host[:, columns]
            self._cat_host[row] = stage._cat_host[:, columns]
            self._action_host[row] = stage._action_host[columns]
            self._player_host[row, 0] = player_id
        if len(updates) < self.bucket:
            self._float_host[len(updates) :] = 0
            self._cat_host[len(updates) :] = 0
            self._action_host[len(updates) :] = 0
            self._player_host[len(updates) :] = 0
        # One upload per packed column type; quantization runs inside the model graph.
        self.floats.copy_(torch.from_numpy(self._float_host))
        self.cats.copy_(torch.from_numpy(self._cat_host))
        self.raw_actions.copy_(torch.from_numpy(self._action_host))
        self.player.copy_(torch.from_numpy(self._player_host))

    def features(self) -> dict[str, Tensor]:
        return packed_features(self.layout, self.floats, self.cats, self.player)
