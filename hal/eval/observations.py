"""Build live policy observations from canonical Dolphin frames."""

from collections.abc import Mapping

from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.representation.observations import flatten_canonical_frame
from hal.sim.inputs import canonical_pre_to_action


def applied_action_from_frame(frame: dict, port: int) -> ControllerAction:
    """Return the observed action in the policy's controller vocabulary."""
    try:
        pre = frame["ports"][port]["leader"]["pre"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"canonical frame has no leader pre-state for port {port}") from error
    action = canonical_pre_to_action(pre)
    return ControllerAction(
        action.main_x,
        action.main_y,
        action.c_x,
        action.c_y,
        action.trigger_l,
        action.trigger_r,
        action.buttons & POLICY_BUTTON_MASK,
    )


def flatten_live_frame(
    frame: dict,
    matchup_characters: Mapping[int, int],
) -> dict[str, float | int]:
    """Add the character-select constants used by the stored replay schema."""
    stage = frame.get("stage")
    if not isinstance(stage, int):
        raise ValueError(f"canonical frame has invalid live stage {stage!r}")
    if set(matchup_characters) != {1, 2}:
        raise ValueError("matchup characters must identify ports 1 and 2")
    return flatten_canonical_frame({**frame, "_matchup": {"stage": stage, "character": matchup_characters}})


def policy_input_from_frame(
    frame: dict,
    *,
    spec: PolicySpec,
    stream_id: int,
    controlled_port: int,
    player_identity: str | None = None,
    desired_return: float | None = 20.0,
    temperature: float = 1.0,
    reset: bool = False,
    flat: Mapping[str, float | int] | None = None,
    matchup_characters: Mapping[int, int] | None = None,
) -> PolicyInput:
    """Project exactly the fields required by one policy backend."""
    if flat is None:
        if matchup_characters is None:
            raise ValueError("matchup characters are required with an unflattened frame")
        values = flatten_live_frame(frame, matchup_characters)
    else:
        values = flat
    return PolicyInput(
        stream_id=stream_id,
        frame_id=int(frame["id"]),
        controlled_port=controlled_port,
        observation={name: values[name] for name in spec.required_observation_fields},
        applied_action=applied_action_from_frame(frame, controlled_port),
        player_identity=player_identity,
        desired_return=desired_return,
        temperature=temperature,
        reset=reset,
    )
