"""Schema-shaped observations for policy preparation and latency checks."""

from dataclasses import replace

import torch

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.representation.features import ACTION_CHANNELS
from hal.representation.features import BASE_PLAYER_PREFIXES
from hal.representation.features import CAT_FEATURES
from hal.representation.features import FLOAT_FEATURES
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import Context
from hal.representation.features import feature_kind
from hal.wire import ITEM_SLOTS
from hal.wire import item_column


def synthetic_context(
    context_length: int,
    batch_size: int,
    device: torch.device,
    *,
    items: bool = True,
) -> Context:
    """Build the current base observation with zeroed optional item slots."""
    if context_length < 1 or batch_size < 1:
        raise ValueError("synthetic context dimensions must be positive")
    features: dict[str, torch.Tensor] = {}
    for prefix in BASE_PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, context_length, device=device)
            features[f"{prefix}_{name}_mask"] = torch.zeros(batch_size, context_length, device=device)
        for name in CAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.zeros(batch_size, context_length, dtype=torch.long, device=device)
    for name in ACTION_CHANNELS:
        features[f"ego_{name}"] = torch.zeros(batch_size, context_length, device=device)
    features["ego_character"] = torch.zeros(batch_size, context_length, dtype=torch.long, device=device)
    features["opp_character"] = torch.zeros(batch_size, context_length, dtype=torch.long, device=device)
    features["stage"] = torch.zeros(batch_size, context_length, dtype=torch.long, device=device)
    if items:
        for slot in range(ITEM_SLOTS):
            for name in ITEM_COLUMNS.cats:
                features[item_column(slot, name)] = torch.zeros(
                    batch_size, context_length, dtype=torch.long, device=device
                )
            for name in ITEM_COLUMNS.floats:
                column = item_column(slot, name)
                features[column] = torch.zeros(batch_size, context_length, device=device)
                features[f"{column}_mask"] = torch.zeros(batch_size, context_length, device=device)
    return Context(
        features=features,
        ctx_pad=torch.zeros(batch_size, dtype=torch.long, device=device),
    )


def canonical_context(ctx: Context, *, items: bool = True) -> Context:
    """Fill absent mask sidecars and stabilize keys before graph capture."""
    features = dict(ctx.features)
    for prefix in BASE_PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            key = f"{prefix}_{name}_mask"
            if key not in features:
                features[key] = torch.zeros_like(features[f"{prefix}_{name}"])
    if items:
        for slot in range(ITEM_SLOTS):
            for name in ITEM_COLUMNS.floats:
                column = item_column(slot, name)
                if f"{column}_mask" not in features:
                    features[f"{column}_mask"] = torch.zeros_like(features[column])
    return replace(ctx, features={name: features[name] for name in sorted(features)})


def make_warmup_observations(
    spec: PolicySpec,
    frame_count: int,
    stream_id: int,
    source_frame: int,
    input_delay_frames: int,
    *,
    reset_first: bool = True,
) -> tuple[PolicyInput, ...]:
    """Build independent frame mappings with the required observation fields."""
    if frame_count < 1 or input_delay_frames < 0:
        raise ValueError("invalid warmup history or input delay")
    first = source_frame - frame_count + 1
    result = []
    for frame in range(first, source_frame + 1):
        observation = {}
        for name in spec.required_observation_fields:
            if name.startswith("p1_"):
                relative = f"ego_{name[3:]}"
            elif name.startswith("p2_"):
                relative = f"opp_{name[3:]}"
            else:
                relative = name
            kind = feature_kind(relative, ITEM_COLUMNS)
            if kind == "drop":
                raise ValueError(f"cannot synthesize unknown observation field {name!r}")
            observation[name] = 0 if kind in ("cat", "button") else 0.0
        result.append(
            PolicyInput(
                stream_id,
                frame,
                1,
                observation,
                NEUTRAL_CONTROLLER_ACTION,
                player_identity="PLATINUM" if spec.requires_player_identity else None,
                reset=reset_first and frame == first,
            )
        )
    return tuple(result)
