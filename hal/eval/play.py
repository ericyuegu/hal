"""Direct netplay loop for the public policy interface."""

import math
import time
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import peppi_py
from peppi_py.game import EndMethod

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import Policy
from hal.inference.api import PolicyInput
from hal.inference.api import RuntimeConfig
from hal.inference.api import validate_policy_inputs
from hal.inference.api import validate_policy_outputs
from hal.inference.transport import ActionTransport
from hal.sim.inputs import canonical_pre_to_action
from hal.sim.inputs import controller_actions_match
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.trajectory import Trajectory
from hal.training.canonical import flatten_canonical_frame


@dataclass(frozen=True, slots=True)
class PlayResult:
    trajectory: Trajectory
    ego_port: int
    opponent_port: int
    wall_seconds: float
    inference_seconds: tuple[float, ...]
    transport_correction_frames: int

    @property
    def inference_p95_ms(self) -> float:
        """Nearest-rank p95 of the complete policy call."""
        if not self.inference_seconds:
            return 0.0
        ordered = sorted(self.inference_seconds)
        index = math.ceil(0.95 * len(ordered)) - 1
        return 1_000.0 * ordered[index]


def _frame_action(frame: dict, port: int) -> ControllerAction:
    try:
        pre = frame["ports"][port]["leader"]["pre"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"canonical frame has no leader pre-state for port {port}") from error
    return canonical_pre_to_action(pre)


def _flat_observation(frame: dict, characters: dict[int, int]) -> dict[str, float | int]:
    stage = frame.get("stage")
    if not isinstance(stage, int):
        raise ValueError(f"canonical frame has invalid live stage {stage!r}")
    return flatten_canonical_frame(
        {
            **frame,
            "_matchup": {
                "stage": stage,
                "character": characters,
            },
        }
    )


def _transport_was_corrected(
    frame: dict,
    port: int,
    expected: ControllerAction,
    recent_scheduled: tuple[ControllerAction, ...],
) -> bool:
    actual = _frame_action(frame, port)
    if controller_actions_match(expected, actual):
        return False
    if any(controller_actions_match(candidate, actual) for candidate in recent_scheduled):
        return True
    raise RuntimeError(
        f"controller transport mismatch at frame {frame.get('id')}: expected {expected!r} "
        f"or one of {len(recent_scheduled)} recent scheduled actions, observed {actual!r}"
    )


def require_completed_replay(replay_dir: Path, previous: Collection[Path]) -> Path:
    """Return this invocation's replay after validating its game-end record."""
    replays = sorted(set(replay_dir.rglob("*.slp")) - set(previous))
    if len(replays) != 1:
        raise RuntimeError(f"expected one new replay in {replay_dir}, found {len(replays)}")
    replay = replays[0]
    try:
        game = peppi_py.read_slippi(str(replay), skip_frames=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"cannot parse completed netplay replay {replay}: {error}") from error
    if game.end is None:
        raise RuntimeError(f"netplay replay has no game-end record: {replay}")
    try:
        method = EndMethod(int(game.end.method))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"netplay replay has unknown game-end method: {replay}") from error
    if method not in (EndMethod.TIME, EndMethod.GAME, EndMethod.RESOLVED):
        raise RuntimeError(f"netplay replay ended via {method.name}: {replay}")
    return replay


def run_netplay_match(
    session: NetplaySession,
    setup: NetplaySetup,
    policy: Policy,
    runtime: RuntimeConfig,
    *,
    player_code: str | None = None,
    max_frames: int = 28_800,
) -> PlayResult:
    """Play one game after flushing menu inputs from Slippi's delay queue."""
    if max_frames < runtime.transport_delay_frames + 3:
        raise ValueError(f"max_frames must be at least transport delay + 3, got {max_frames}")
    if session.online_delay != runtime.transport_delay_frames:
        raise ValueError(
            f"session delay {session.online_delay} differs from policy delay {runtime.transport_delay_frames}"
        )
    first_frame = session.start_match(setup)
    if session.ego_port is None or session.opponent_port is None:
        raise RuntimeError("netplay ports were not discovered")
    ego_port = session.ego_port
    opponent_port = session.opponent_port
    characters = {port: int(first_frame["ports"][port]["leader"]["post"]["character"]) for port in (1, 2)}
    captured = [first_frame]
    transport = ActionTransport(runtime.transport_delay_frames)
    started = time.monotonic()

    # Menu navigation can leave actions in Dolphin's queue. A Slippi time-sync
    # stall can hold one sample, so flush until a neutral state is observed.
    current = first_frame
    for flush_index in range(runtime.transport_delay_frames + 120):
        transport.submit(NEUTRAL_CONTROLLER_ACTION)
        current, in_game = session.step(NEUTRAL_CONTROLLER_ACTION)
        captured.append(current)
        if not in_game:
            raise RuntimeError("netplay left live play while flushing menu inputs")
        if flush_index >= runtime.transport_delay_frames and controller_actions_match(
            NEUTRAL_CONTROLLER_ACTION,
            _frame_action(current, ego_port),
        ):
            break
    else:
        raise RuntimeError("netplay did not reach a neutral controller state after menu navigation")

    inference_seconds: list[float] = []
    transport_correction_frames = 0
    # Slippi 3.6.4 can replay a recent local pad during its 30-frame clock
    # correction. Its rollback window is seven frames, so retain the current
    # scheduled action and the seven that precede it.
    recent_scheduled: deque[ControllerAction] = deque(
        (NEUTRAL_CONTROLLER_ACTION,),
        maxlen=8,
    )
    first_policy_frame = True
    while len(captured) < max_frames:
        item = PolicyInput(
            stream_id=0,
            frame_id=int(current["id"]),
            controlled_port=ego_port,
            observation=_flat_observation(current, characters),
            applied_action=_frame_action(current, ego_port),
            pending_actions=transport.pending,
            player_code=player_code,
            reset=first_policy_frame,
        )
        validate_policy_inputs(policy.spec, runtime, (item,))
        inference_started = time.perf_counter()
        outputs = tuple(policy.step((item,)))
        inference_seconds.append(time.perf_counter() - inference_started)
        action = validate_policy_outputs((item,), outputs)[0]
        due = transport.submit(action)
        recent_scheduled.append(due)
        current, in_game = session.step(action)
        captured.append(current)
        if in_game:
            transport_correction_frames += _transport_was_corrected(
                current,
                ego_port,
                due,
                tuple(recent_scheduled),
            )
        if not in_game:
            break
        first_policy_frame = False
    else:
        raise RuntimeError(f"netplay game did not finish within {max_frames} frames")

    return PlayResult(
        trajectory=Trajectory.from_capture(captured, (1, 2)),
        ego_port=ego_port,
        opponent_port=opponent_port,
        wall_seconds=time.monotonic() - started,
        inference_seconds=tuple(inference_seconds),
        transport_correction_frames=transport_correction_frames,
    )
