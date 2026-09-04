"""Discrete controller vocabulary shared by training and inference."""

from typing import Final
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hal.training import scoring
from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM

CONTROLLER_GROUP_NAMES: Final[tuple[str, ...]] = (
    "buttons",
    "main_stick",
    "c_stick",
    "triggers",
)
CONTROLLER_GROUP_VOCABS: Final[tuple[int, ...]] = (
    scoring.N_BUTTON_COMBOS,
    scoring.STICK_CLUSTER_CENTERS_MAIN.shape[0],
    scoring.STICK_CLUSTER_CENTERS_C.shape[0],
    scoring.TRIGGER_CENTERS.shape[0] ** 2,
)
CONTROLLER_GROUP_COUNT: Final[int] = len(CONTROLLER_GROUP_NAMES)
BUTTONS_GROUP, MAIN_STICK_GROUP, C_STICK_GROUP, TRIGGERS_GROUP = range(CONTROLLER_GROUP_COUNT)
CONTROLLER_GROUP_INDEX: Final[dict[str, int]] = {name: index for index, name in enumerate(CONTROLLER_GROUP_NAMES)}
CONTROLLER_DECODE_ORDER: Final[tuple[str, ...]] = (
    "c_stick",
    "main_stick",
    "triggers",
    "buttons",
)

_CONTINUOUS_CHANNEL_COUNT = 6
_TRIGGER_LEFT_CHANNEL = ACTION_CHANNELS.index("trigger_l")
_TRIGGER_RIGHT_CHANNEL = ACTION_CHANNELS.index("trigger_r")
_BUTTON_LEFT_CHANNEL = ACTION_CHANNELS.index("button_l")
_BUTTON_RIGHT_CHANNEL = ACTION_CHANNELS.index("button_r")


def _rms_norm(values: Tensor) -> Tensor:
    return F.rms_norm(values, (values.shape[-1],), eps=1e-6)


class DiscreteControllerCodec(nn.Module):
    """Map the raw controller wire to factorized categorical tokens."""

    main_centers: Tensor
    c_centers: Tensor
    trigger_centers: Tensor
    button_valid_for_trigger: Tensor

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.class_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(CONTROLLER_GROUP_VOCABS[CONTROLLER_GROUP_INDEX[name]], embed_dim)
                for name in CONTROLLER_GROUP_NAMES
            }
        )
        semantic_dims = {"buttons": 8, "main_stick": 2, "c_stick": 2, "triggers": 2}
        self.semantic_projections = nn.ModuleDict(
            {name: nn.Linear(width, embed_dim, bias=False) for name, width in semantic_dims.items()}
        )
        self.register_buffer("main_centers", scoring.STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", scoring.STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trigger_centers", scoring.TRIGGER_CENTERS.clone())
        button_bits = scoring.combo_to_buttons(torch.arange(CONTROLLER_GROUP_VOCABS[BUTTONS_GROUP]))
        trigger_pairs = torch.arange(CONTROLLER_GROUP_VOCABS[TRIGGERS_GROUP])
        trigger_count = len(self.trigger_centers)
        left_full = trigger_pairs.div(trigger_count, rounding_mode="floor") == trigger_count - 1
        right_full = trigger_pairs.remainder(trigger_count) == trigger_count - 1
        left_click = button_bits[:, _BUTTON_LEFT_CHANNEL - _CONTINUOUS_CHANNEL_COUNT].bool()
        right_click = button_bits[:, _BUTTON_RIGHT_CHANNEL - _CONTINUOUS_CHANNEL_COUNT].bool()
        valid = (~left_click[None, :] | left_full[:, None]) & (~right_click[None, :] | right_full[:, None])
        self.register_buffer("button_valid_for_trigger", valid)

    def _class_embedding(self, name: str) -> nn.Embedding:
        return cast(nn.Embedding, self.class_embeddings[name])

    def _semantic_projection(self, name: str) -> nn.Linear:
        return cast(nn.Linear, self.semantic_projections[name])

    @staticmethod
    def canonicalize(actions: Tensor) -> Tensor:
        if actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"controller actions must end in {ACTION_DIM} channels, got {tuple(actions.shape)}")
        out = actions.clone()
        out[..., _TRIGGER_LEFT_CHANNEL] = torch.where(
            out[..., _BUTTON_LEFT_CHANNEL] > 0.5,
            torch.ones_like(out[..., _TRIGGER_LEFT_CHANNEL]),
            out[..., _TRIGGER_LEFT_CHANNEL],
        )
        out[..., _TRIGGER_RIGHT_CHANNEL] = torch.where(
            out[..., _BUTTON_RIGHT_CHANNEL] > 0.5,
            torch.ones_like(out[..., _TRIGGER_RIGHT_CHANNEL]),
            out[..., _TRIGGER_RIGHT_CHANNEL],
        )
        return out

    def quantize(self, actions: Tensor) -> Tensor:
        actions = self.canonicalize(actions)
        continuous = actions[..., :_CONTINUOUS_CHANNEL_COUNT]
        buttons_raw = actions[..., _CONTINUOUS_CHANNEL_COUNT:]
        buttons = scoring.buttons_to_combo(buttons_raw)
        main = scoring.nearest_cluster(continuous[..., 0:2], self.main_centers)
        c_stick = scoring.nearest_cluster(continuous[..., 2:4], self.c_centers)
        trigger_pair = scoring.nearest_center(continuous[..., 4:6], self.trigger_centers)
        triggers = trigger_pair[..., 0] * self.trigger_centers.shape[0] + trigger_pair[..., 1]
        return torch.stack((buttons, main, c_stick, triggers), dim=-1)

    def dequantize(self, indices: Tensor) -> Tensor:
        trigger_count = self.trigger_centers.shape[0]
        buttons = scoring.combo_to_buttons(indices[..., BUTTONS_GROUP])
        main = scoring.cluster_to_xy(indices[..., MAIN_STICK_GROUP], self.main_centers)
        c_stick = scoring.cluster_to_xy(indices[..., C_STICK_GROUP], self.c_centers)
        trigger_left = scoring.center_to_value(
            indices[..., TRIGGERS_GROUP] // trigger_count,
            self.trigger_centers,
        )
        trigger_right = scoring.center_to_value(
            indices[..., TRIGGERS_GROUP] % trigger_count,
            self.trigger_centers,
        )
        return torch.cat(
            (main, c_stick, torch.stack((trigger_left, trigger_right), dim=-1), buttons),
            dim=-1,
        )

    def semantic_values(self, name: str, indices: Tensor) -> Tensor:
        if name == "buttons":
            return scoring.combo_to_buttons(indices).to(self._class_embedding(name).weight.dtype)
        if name == "main_stick":
            return scoring.cluster_to_xy(indices, self.main_centers)
        if name == "c_stick":
            return scoring.cluster_to_xy(indices, self.c_centers)
        if name == "triggers":
            trigger_count = self.trigger_centers.shape[0]
            return torch.stack(
                (
                    scoring.center_to_value(indices // trigger_count, self.trigger_centers),
                    scoring.center_to_value(indices % trigger_count, self.trigger_centers),
                ),
                dim=-1,
            )
        raise ValueError(f"unknown controller group {name!r}")

    def group_embedding(self, name: str, indices: Tensor) -> Tensor:
        class_embedding = self._class_embedding(name)
        semantic = self.semantic_values(name, indices).to(class_embedding.weight.dtype)
        return _rms_norm(class_embedding(indices) + self._semantic_projection(name)(semantic))

    def embed_groups(self, indices: Tensor) -> dict[str, Tensor]:
        return {
            name: self.group_embedding(name, indices[..., CONTROLLER_GROUP_INDEX[name]])
            for name in CONTROLLER_GROUP_NAMES
        }

    def embed_frame(self, indices: Tensor, embedded: dict[str, Tensor] | None = None) -> Tensor:
        values = self.embed_groups(indices) if embedded is None else embedded
        return torch.cat([values[name] for name in CONTROLLER_GROUP_NAMES], dim=-1)

    def button_mask(self, trigger_indices: Tensor) -> Tensor:
        return ~self.button_valid_for_trigger[trigger_indices]
