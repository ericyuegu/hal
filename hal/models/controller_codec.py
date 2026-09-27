"""Discrete controller vocabulary shared by training and inference."""

from typing import Final
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM

# --- joint-2D stick clusters -------------------------------------------------
# Hand-tuned (x, y) stick targets in the action [-1, 1] space: neutral, partial/full
# tilts, cardinals, wavedash & ledgedash angles (17/30/45/60/72.5 deg), shield-drop and
# angled-tilt diagonals. A *joint* categorical over correlated x/y, unlike per-axis bins.
#
# Main-stick mass is 34% neutral and ~90% of the rest at magnitude >=0.95 (the rim), where
# skill-critical angles live (DI, wavedash, firefox). The rim spokes below are spaced to
# ~8-10deg so no human input lands far from a center. Inner pose centers (tilts, partial
# deflections) are kept for the off-rim minority.
#
# Layout (65 centers): neutral + two cardinal-tilt magnitudes + two intermediate rings
# (r~0.5, r~0.85, 8 spokes each — these mid-magnitude gaps held most of the >0.1 error) +
# a 40-spoke rim at 9deg. All points snapped to the 1/80 stick grid.
_STICK_CLUSTER_XY: tuple[tuple[float, float], ...] = (
    # neutral
    (0.0, 0.0),
    # cardinal tilts (jab/tilt range)
    (0.35, 0.0),
    (-0.35, 0.0),
    (0.0, 0.35),
    (0.0, -0.35),
    (0.675, 0.0),
    (-0.675, 0.0),
    (0.0, 0.675),
    (0.0, -0.675),
    # inner ring r~0.5 (angled tilts, shield-drop)
    (0.5, 0.0),
    (0.35, 0.35),
    (0.0, 0.5),
    (-0.35, 0.35),
    (-0.5, 0.0),
    (-0.35, -0.35),
    (0.0, -0.5),
    (0.35, -0.35),
    # outer ring r~0.85 (between the inner ring and the rim)
    (0.85, 0.0),
    (0.6, 0.6),
    (0.0, 0.85),
    (-0.6, 0.6),
    (-0.85, 0.0),
    (-0.6, -0.6),
    (0.0, -0.85),
    (0.6, -0.6),
    # full rim, 40 spokes @ 9deg (magnitude ~1.0; cos/sin snapped to the 1/80 grid). The
    # skill-critical angles (DI, wavedash, firefox) live here, hence the dense spacing.
    (1.0, 0.0),  # 0deg
    (0.9875, 0.1625),
    (0.95, 0.3125),
    (0.8875, 0.45),
    (0.8125, 0.5875),
    (0.7125, 0.7125),  # ~45
    (0.5875, 0.8125),
    (0.45, 0.8875),
    (0.3125, 0.95),
    (0.1625, 0.9875),
    (0.0, 1.0),  # 90
    (-0.1625, 0.9875),
    (-0.3125, 0.95),
    (-0.45, 0.8875),
    (-0.5875, 0.8125),
    (-0.7125, 0.7125),  # ~135
    (-0.8125, 0.5875),
    (-0.8875, 0.45),
    (-0.95, 0.3125),
    (-0.9875, 0.1625),
    (-1.0, 0.0),  # 180
    (-0.9875, -0.1625),
    (-0.95, -0.3125),
    (-0.8875, -0.45),
    (-0.8125, -0.5875),
    (-0.7125, -0.7125),  # ~225
    (-0.5875, -0.8125),
    (-0.45, -0.8875),
    (-0.3125, -0.95),
    (-0.1625, -0.9875),
    (0.0, -1.0),  # 270
    (0.1625, -0.9875),
    (0.3125, -0.95),
    (0.45, -0.8875),
    (0.5875, -0.8125),
    (0.7125, -0.7125),  # ~315
    (0.8125, -0.5875),
    (0.8875, -0.45),
    (0.95, -0.3125),
    (0.9875, -0.1625),
)

STICK_CLUSTER_CENTERS_MAIN: Tensor = torch.tensor(_STICK_CLUSTER_XY, dtype=torch.float32)

# C-stick is 95% exact-neutral and its non-neutral mass sits overwhelmingly on the rim at the
# cardinals/diagonals (smash attacks); a dense pose set would leave most clusters empty. Its
# own 9-point set: neutral + four full cardinals + four full diagonals.
_C_STICK_XY: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.7, 0.7),
    (-0.7, 0.7),
    (0.7, -0.7),
    (-0.7, -0.7),
)
STICK_CLUSTER_CENTERS_C: Tensor = torch.tensor(_C_STICK_XY, dtype=torch.float32)


def nearest_cluster(xy: Tensor, centers: Tensor) -> Tensor:
    """``[..., 2]`` stick coords → ``[...]`` index of the nearest (L2) cluster center."""
    c = centers.to(xy.device)
    return (xy.unsqueeze(-2) - c).pow(2).sum(-1).argmin(-1)


def cluster_to_xy(idx: Tensor, centers: Tensor) -> Tensor:
    """Inverse of ``nearest_cluster``: ``[...]`` indices → ``[..., 2]`` center coords."""
    return centers.to(idx.device)[idx]


# --- 1D trigger centers ------------------------------------------------------
# Triggers are 92.6% exactly 0, 5.9% exactly 1.0, ~1.4% in a broad analog band (p10 0.34,
# p90 0.93); everything below the extract deadzone (43/140 ~ 0.307) is 0 by construction.
# A hand-tuned 5-center set spends its mass where the data is — the two spikes plus three
# analog-band points — instead of the empty low bins a uniform grid wastes.
TRIGGER_CENTERS: Tensor = torch.tensor((0.0, 0.35, 0.6, 0.85, 1.0), dtype=torch.float32)


def nearest_center(x: Tensor, centers: Tensor) -> Tensor:
    """``[...]`` scalar values → ``[...]`` index of the nearest (1D L1) center. The 1D analog
    of ``nearest_cluster`` for per-shoulder triggers."""
    c = centers.to(x.device)
    return (x.unsqueeze(-1) - c).abs().argmin(-1)


def center_to_value(idx: Tensor, centers: Tensor) -> Tensor:
    """Inverse of ``nearest_center``: ``[...]`` indices → ``[...]`` center values."""
    return centers.to(idx.device)[idx]


N_BUTTONS = 8
N_BUTTON_COMBOS = 1 << N_BUTTONS


def buttons_to_combo(buttons: Tensor) -> Tensor:
    """``[..., 8]`` button bits {0,1} → ``[...]`` long combo id in ``[0, 256)``. Bit ``k`` of
    the id is button channel ``k`` (ACTION_CHANNELS order), so the full co-press product is
    representable and conflicting presses are impossible by construction."""
    bits = (buttons > 0.5).long()
    weights = (1 << torch.arange(N_BUTTONS, device=buttons.device)).long()
    return (bits * weights).sum(-1)


def combo_to_buttons(combo: Tensor) -> Tensor:
    """Inverse of ``buttons_to_combo``: ``[...]`` combo id → ``[..., 8]`` float bits {0,1}."""
    bit = torch.arange(N_BUTTONS, device=combo.device)
    return ((combo.unsqueeze(-1) >> bit) & 1).float()


CONTROLLER_GROUP_NAMES: Final[tuple[str, ...]] = (
    "buttons",
    "main_stick",
    "c_stick",
    "triggers",
)
CONTROLLER_GROUP_VOCABS: Final[tuple[int, ...]] = (
    N_BUTTON_COMBOS,
    STICK_CLUSTER_CENTERS_MAIN.shape[0],
    STICK_CLUSTER_CENTERS_C.shape[0],
    TRIGGER_CENTERS.shape[0] ** 2,
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

CONTINUOUS_CHANNEL_COUNT: Final[int] = 6
TRIGGER_LEFT_CHANNEL: Final[int] = ACTION_CHANNELS.index("trigger_l")
TRIGGER_RIGHT_CHANNEL: Final[int] = ACTION_CHANNELS.index("trigger_r")
BUTTON_LEFT_CHANNEL: Final[int] = ACTION_CHANNELS.index("button_l")
BUTTON_RIGHT_CHANNEL: Final[int] = ACTION_CHANNELS.index("button_r")


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
        self.register_buffer("main_centers", STICK_CLUSTER_CENTERS_MAIN.clone())
        self.register_buffer("c_centers", STICK_CLUSTER_CENTERS_C.clone())
        self.register_buffer("trigger_centers", TRIGGER_CENTERS.clone())
        button_bits = combo_to_buttons(torch.arange(CONTROLLER_GROUP_VOCABS[BUTTONS_GROUP]))
        trigger_pairs = torch.arange(CONTROLLER_GROUP_VOCABS[TRIGGERS_GROUP])
        trigger_count = len(self.trigger_centers)
        left_full = trigger_pairs.div(trigger_count, rounding_mode="floor") == trigger_count - 1
        right_full = trigger_pairs.remainder(trigger_count) == trigger_count - 1
        left_click = button_bits[:, BUTTON_LEFT_CHANNEL - CONTINUOUS_CHANNEL_COUNT].bool()
        right_click = button_bits[:, BUTTON_RIGHT_CHANNEL - CONTINUOUS_CHANNEL_COUNT].bool()
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
        out[..., TRIGGER_LEFT_CHANNEL] = torch.where(
            out[..., BUTTON_LEFT_CHANNEL] > 0.5,
            torch.ones_like(out[..., TRIGGER_LEFT_CHANNEL]),
            out[..., TRIGGER_LEFT_CHANNEL],
        )
        out[..., TRIGGER_RIGHT_CHANNEL] = torch.where(
            out[..., BUTTON_RIGHT_CHANNEL] > 0.5,
            torch.ones_like(out[..., TRIGGER_RIGHT_CHANNEL]),
            out[..., TRIGGER_RIGHT_CHANNEL],
        )
        return out

    def quantize(self, actions: Tensor) -> Tensor:
        actions = self.canonicalize(actions)
        continuous = actions[..., :CONTINUOUS_CHANNEL_COUNT]
        buttons_raw = actions[..., CONTINUOUS_CHANNEL_COUNT:]
        buttons = buttons_to_combo(buttons_raw)
        main = nearest_cluster(continuous[..., 0:2], self.main_centers)
        c_stick = nearest_cluster(continuous[..., 2:4], self.c_centers)
        trigger_pair = nearest_center(continuous[..., 4:6], self.trigger_centers)
        triggers = trigger_pair[..., 0] * self.trigger_centers.shape[0] + trigger_pair[..., 1]
        return torch.stack((buttons, main, c_stick, triggers), dim=-1)

    def dequantize(self, indices: Tensor) -> Tensor:
        trigger_count = self.trigger_centers.shape[0]
        buttons = combo_to_buttons(indices[..., BUTTONS_GROUP])
        main = cluster_to_xy(indices[..., MAIN_STICK_GROUP], self.main_centers)
        c_stick = cluster_to_xy(indices[..., C_STICK_GROUP], self.c_centers)
        trigger_left = center_to_value(
            indices[..., TRIGGERS_GROUP] // trigger_count,
            self.trigger_centers,
        )
        trigger_right = center_to_value(
            indices[..., TRIGGERS_GROUP] % trigger_count,
            self.trigger_centers,
        )
        return torch.cat(
            (main, c_stick, torch.stack((trigger_left, trigger_right), dim=-1), buttons),
            dim=-1,
        )

    def semantic_values(self, name: str, indices: Tensor) -> Tensor:
        if name == "buttons":
            return combo_to_buttons(indices).to(self._class_embedding(name).weight.dtype)
        if name == "main_stick":
            return cluster_to_xy(indices, self.main_centers)
        if name == "c_stick":
            return cluster_to_xy(indices, self.c_centers)
        if name == "triggers":
            trigger_count = self.trigger_centers.shape[0]
            return torch.stack(
                (
                    center_to_value(indices // trigger_count, self.trigger_centers),
                    center_to_value(indices % trigger_count, self.trigger_centers),
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
