"""Action-state event counts shared by PPO rollouts and held-out evaluation."""

from dataclasses import dataclass

import melee
import numpy as np
from numpy.typing import NDArray

from hal.data.behavior import EDGE_X
from hal.data.behavior import INACTIVE_ACTIONS
from hal.data.behavior import SHIELD_ACTIONS

_ACTION = melee.Action
EVENT_STATES: dict[str, frozenset[int]] = {
    "forward_smash": frozenset(range(_ACTION.FSMASH_HIGH.value, _ACTION.FSMASH_LOW.value + 1)),
    "up_smash": frozenset({_ACTION.UPSMASH.value}),
    "down_smash": frozenset({_ACTION.DOWNSMASH.value}),
    "neutral_air": frozenset({_ACTION.NAIR.value}),
    "forward_air": frozenset({_ACTION.FAIR.value}),
    "back_air": frozenset({_ACTION.BAIR.value}),
    "up_air": frozenset({_ACTION.UAIR.value}),
    "down_air": frozenset({_ACTION.DAIR.value}),
    "grab": frozenset({_ACTION.GRAB.value, _ACTION.GRAB_RUNNING.value}),
    "throw": frozenset(range(_ACTION.THROW_FORWARD.value, _ACTION.THROW_DOWN.value + 1)),
    "shield": SHIELD_ACTIONS,
    "roll": frozenset({_ACTION.ROLL_FORWARD.value, _ACTION.ROLL_BACKWARD.value}),
    "spot_dodge": frozenset({_ACTION.SPOTDODGE.value}),
    "tech": frozenset(
        {
            _ACTION.NEUTRAL_TECH.value,
            _ACTION.FORWARD_TECH.value,
            _ACTION.BACKWARD_TECH.value,
            _ACTION.WALL_TECH.value,
            _ACTION.WALL_TECH_JUMP.value,
            _ACTION.CEILING_TECH.value,
        }
    ),
    "ledge_grab": frozenset({_ACTION.EDGE_CATCHING.value}),
}
EVENT_STATES["smash"] = frozenset().union(
    *(EVENT_STATES[name] for name in ("forward_smash", "up_smash", "down_smash"))
)
EVENT_STATES["aerial"] = frozenset().union(
    *(EVENT_STATES[name] for name in ("neutral_air", "forward_air", "back_air", "up_air", "down_air"))
)
COUNT_NAMES = (*EVENT_STATES, "zero_damage_self_destruct", "active_frames", "offstage_frames")
_INACTIVE = tuple(INACTIVE_ACTIONS)
_EDGES = np.full(256, np.nan)
for _stage, _edge in EDGE_X.items():
    _EDGES[_stage.value] = _edge


@dataclass(frozen=True, slots=True)
class BehaviorCounts:
    counts: NDArray[np.int64]

    def metrics(self, role: str) -> dict[str, float]:
        result = {
            f"behavior/{role}_{name}_count": float(value) for name, value in zip(COUNT_NAMES, self.counts, strict=True)
        }
        active = int(self.counts[-2])
        if active:
            result.update(
                {
                    f"behavior/{role}_{name}_per_min": float(value) * 3600 / active
                    for name, value in zip(COUNT_NAMES[:-2], self.counts[:-2], strict=True)
                }
            )
            result[f"behavior/{role}_offstage_fraction"] = float(self.counts[-1]) / active
        return result


class BehaviorTracker:
    """Retain stock damage and action transitions across rollout boundaries.

    Counts use entries into each state set, including shield episodes. Exposure
    excludes death, respawn and entry states for either player. A zero-damage
    self-destruct is a stock loss with no positive percent since respawn; it is
    an explicit proxy because native observations have no KO-credit field.
    """

    def __init__(self, players: int) -> None:
        self.state = np.zeros((players, 3), dtype=np.int64)
        self.state[:, 0] = -1
        self.state[:, 1] = 4
        self.counts = np.zeros((players, len(COUNT_NAMES)), dtype=np.int64)

    def reset(self, streams: NDArray[np.int64]) -> None:
        self.state[streams] = (-1, 4, 0)

    def restore(self, state: NDArray[np.int64]) -> None:
        if (
            not isinstance(state, np.ndarray)
            or state.dtype != np.int64
            or state.shape != self.state.shape
            or np.any((state[:, 0] < -1) | (state[:, 0] >= 1024))
            or np.any((state[:, 1] < 0) | (state[:, 1] > 4))
            or np.any((state[:, 2:] < 0) | (state[:, 2:] > 1))
        ):
            raise ValueError("invalid behavior tracker state")
        self.state[:] = state

    def observe(
        self,
        streams: NDArray[np.int64],
        action: NDArray[np.int64],
        stock: NDArray[np.int64],
        percent: NDArray[np.float64],
        x: NDArray[np.float64],
        stage: NDArray[np.int64],
    ) -> None:
        if action.shape != (len(streams) // 2, 2) or any(value.shape != action.shape for value in (stock, percent, x)):
            raise ValueError("behavior columns must contain both ports of each match")
        if np.any((stage < 0) | (stage >= len(_EDGES))) or not np.isfinite(_EDGES[stage]).all():
            raise ValueError("behavior stage has no walkable edge")
        previous = self.state[streams]
        flat = action.ravel()
        active = (~np.isin(action, _INACTIVE) & (action >= 0) & np.isfinite(x)).all(-1).repeat(2)
        for column, states in enumerate(EVENT_STATES.values()):
            entered = np.isin(flat, tuple(states)) & ~np.isin(previous[:, 0], tuple(states))
            self.counts[streams, column] += entered & active
        damage = previous[:, 2].astype(bool) | ((percent.ravel() > 0) & ~np.isin(flat, _INACTIVE))
        lost = stock.ravel() < previous[:, 1]
        self.counts[streams, -3] += lost & ~damage
        self.counts[streams, -2] += active
        self.counts[streams, -1] += active & (np.abs(x.ravel()) > _EDGES[stage].repeat(2))
        self.state[streams, 0] = flat
        self.state[streams, 1] = stock.ravel()
        self.state[streams, 2] = damage & ~lost


def sequence_counts(
    action: NDArray[np.int64],
    stock: NDArray[np.int64],
    percent: NDArray[np.float64],
    x: NDArray[np.float64],
    stage: int,
) -> tuple[BehaviorCounts, BehaviorCounts]:
    """Count a fresh two-player match without a Python loop over frames."""
    edge = EDGE_X[melee.Stage(stage)]
    active = (~np.isin(action, _INACTIVE) & (action >= 0) & np.isfinite(x)).all(-1)
    counts = np.zeros((2, len(COUNT_NAMES)), dtype=np.int64)
    previous = np.concatenate((np.full((1, 2), -1), action[:-1]))
    for column, states in enumerate(EVENT_STATES.values()):
        counts[:, column] = (np.isin(action, tuple(states)) & ~np.isin(previous, tuple(states)) & active[:, None]).sum(
            0
        )
    for side in (0, 1):
        lost = np.flatnonzero(np.diff(np.concatenate(([4], stock[:, side]))) < 0)
        start = 0
        for at in lost:
            touched = (percent[start : at + 1, side] > 0) & ~np.isin(action[start : at + 1, side], _INACTIVE)
            counts[side, -3] += not touched.any()
            start = int(at) + 1
    counts[:, -2] = active.sum()
    counts[:, -1] = ((np.abs(x) > edge) & active[:, None]).sum(0)
    return BehaviorCounts(counts[0]), BehaviorCounts(counts[1])
