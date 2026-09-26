"""Schema-shaped observations for policy preparation and latency checks."""

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.training.features import ITEM_COLUMNS
from hal.training.features import feature_kind


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
                (NEUTRAL_CONTROLLER_ACTION,) * input_delay_frames,
                player_identity="PLATINUM" if spec.requires_player_identity else None,
                reset=reset_first and frame == first,
            )
        )
    return tuple(result)
