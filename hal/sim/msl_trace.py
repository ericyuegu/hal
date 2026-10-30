"""Convert native observations to the MSLTRACE1 viewer format without Torch.

Native observations have no action-frame counter. ``actionFrame`` is the
zero-based count since the action id changed; it cannot detect same-id restarts.
Unavailable player fields use the viewer's zero defaults. Items and stage rows
are omitted. Character ids already use MSL's internal ids, as in native.py.
"""

import numpy as np
from numpy.typing import NDArray

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.sim import native
from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM
from hal.wire import BUTTON_BITS
from hal.wire import MASK_INT32

PLAYER_FIELDS = (
    "charId",
    "actionId",
    "actionFrame",
    "x",
    "y",
    "facing",
    "grounded",
    "percent",
    "shield",
    "stocks",
    "jumps",
    "hitlag",
    "hurtbox",
    "dead",
)
INPUT_FIELDS = ("buttons", "mainX", "mainY", "cX", "cY", "l", "r")


def column_values(columns: NDArray[np.uint32], name: str) -> NDArray:
    """Decode a native word column using the stored numeric schema."""
    if columns.dtype != np.uint32 or columns.ndim != 2 or columns.shape[1] != len(native.OBSERVATION_FIELDS):
        raise ValueError("native columns must be uint32[T,71]")
    dtype = np.float32 if np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "f" else np.int32
    return columns[:, native.OBSERVATION_FIELDS.index(name)].copy().view(dtype)


def to_msltrace(columns: NDArray[np.uint32], applied_actions: NDArray[np.float32]) -> dict[str, object]:
    """Return dense MSLTRACE1 keyframes, including both players' applied inputs."""
    stages = column_values(columns, "stage")
    count = len(columns)
    if count == 0 or len(np.unique(stages)) != 1 or int(stages[0]) not in native._STAGE_TO_NATIVE:
        raise ValueError("trace needs one supported native stage and at least one frame")
    if (
        applied_actions.dtype != np.float32
        or applied_actions.shape != (count, 2, ACTION_DIM)
        or not np.isfinite(applied_actions).all()
        or np.any(np.abs(applied_actions[..., :4]) > 1)
        or np.any((applied_actions[..., 4:] < 0) | (applied_actions[..., 4:] > 1))
    ):
        raise ValueError("applied actions must be finite native float32[T,2,14] inputs")
    characters = [column_values(columns, f"p{side}_character") for side in (1, 2)]
    for character in characters:
        if len(np.unique(character)) != 1 or not 0 <= int(character[0]) <= 26 or int(character[0]) == 11:
            raise ValueError("trace needs constant supported native characters")
    players: list[list[list[float | int]]] = []
    inputs: list[list[list[object]]] = []
    for side in range(2):
        prefix = f"p{side + 1}_"
        values = {
            name: column_values(columns, prefix + name)
            for name in (
                "action",
                "position_x",
                "position_y",
                "direction",
                "airborne",
                "percent",
                "shield",
                "stock",
                "jumps_used",
                "hitlag_left",
                "hurtbox_state",
            )
        }
        if any(np.isinf(value).any() for value in values.values()):
            raise ValueError("native player fields contain infinity")
        rows: list[list[float | int]] = []
        input_rows: list[list[object]] = []
        age = 0
        for frame in range(count):
            age = age + 1 if frame and values["action"][frame] == values["action"][frame - 1] else 0
            available = {
                name: float(value[frame]) if np.isfinite(value[frame]) and value[frame] != MASK_INT32 else 0.0
                for name, value in values.items()
            }
            rows.append(
                [
                    int(characters[side][frame]),
                    int(available["action"]),
                    age,
                    available["position_x"],
                    available["position_y"],
                    available["direction"],
                    int(values["airborne"][frame] == 0),
                    available["percent"],
                    available["shield"],
                    int(available["stock"]),
                    int(available["jumps_used"]),  # Native uses this historical name for jumps left.
                    available["hitlag_left"],
                    int(available["hurtbox_state"]),
                    int(available["stock"] == 0),
                ]
            )
            action = applied_actions[frame, side]
            buttons = sum(
                BUTTON_BITS[name.removeprefix("button_")]
                for name, value in zip(ACTION_CHANNELS[6:], action[6:], strict=True)
                if value > 0.5
            )
            input_rows.append([0, frame, [buttons, *map(float, action[:4]), *map(float, action[4:6])]])
        players.append(rows)
        inputs.append(input_rows)
    return {
        "format": "MSLTRACE1",
        "schemaVersion": 1,
        "match": {
            "stageId": native._STAGE_TO_NATIVE[int(stages[0])],
            "numPlayers": 2,
            "isTeams": False,
            "players": [{"charId": int(c[0]), "port": i + 1} for i, c in enumerate(characters)],
        },
        "frames": {
            "encoding": "sparse-delta-v1",
            "playerFields": PLAYER_FIELDS,
            "rows": [[0, frame, 0, [players[0][frame], players[1][frame]]] for frame in range(count)],
        },
        "inputs": {"encoding": "sparse-delta-v1", "fields": INPUT_FIELDS, "players": inputs},
        "metadata": {"actionFrame": "Approximate: frames since action id changed; same-id restarts are unknown."},
    }
