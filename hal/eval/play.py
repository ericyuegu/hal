"""Direct netplay loop for the public policy interface."""

import math
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import peppi_py
from loguru import logger
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
from hal.wire import BUTTON_BITS


@dataclass(frozen=True, slots=True)
class PlayResult:
    trajectory: Trajectory
    ego_port: int
    opponent_port: int
    stage: int
    wall_seconds: float
    inference_seconds: tuple[float, ...]
    transport_correction_frames: int

    @property
    def inference_p95_ms(self) -> float:
        """Nearest-rank p95 of the complete policy call."""
        return _p95_ms(self.inference_seconds)


def _p95_ms(values: Collection[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return 1_000.0 * ordered[math.ceil(0.95 * len(ordered)) - 1]


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


def _action_summary(action: ControllerAction) -> str:
    buttons = "+".join(name.upper() for name, bit in BUTTON_BITS.items() if action.buttons & bit) or "-"
    return (
        f"main=({action.main_x:+.2f},{action.main_y:+.2f}) "
        f"c=({action.c_x:+.2f},{action.c_y:+.2f}) "
        f"triggers=({action.trigger_l:.2f},{action.trigger_r:.2f}) buttons={buttons}"
    )


def _log_match_progress(
    frame: dict,
    *,
    ego_port: int,
    opponent_port: int,
    submitted: ControllerAction,
    expected: ControllerAction,
    observed: ControllerAction,
    inference_seconds: Collection[float],
    correction_frames: int,
    changed_frames: int,
    button_edges: int,
    main_stick_changes: int,
    c_stick_changes: int,
    started: float,
) -> None:
    ego = frame["ports"][ego_port]["leader"]["post"]
    opponent = frame["ports"][opponent_port]["leader"]["post"]
    logger.info(
        "netplay progress frame={} wall={:.1f}s stocks={}:{} percent={:.1f}:{:.1f} "
        "position=({:.1f},{:.1f}):({:.1f},{:.1f}) policy_p95={:.1f}ms corrections={} "
        "output_changes={}/{} button_edges={} main_changes={} c_changes={}\n"
        "  submitted: {}\n  expected:  {}\n  observed:  {}",
        frame["id"],
        time.monotonic() - started,
        ego["stock"],
        opponent["stock"],
        ego["percent"],
        opponent["percent"],
        ego["position"]["x"],
        ego["position"]["y"],
        opponent["position"]["x"],
        opponent["position"]["y"],
        _p95_ms(inference_seconds),
        correction_frames,
        changed_frames,
        max(0, len(inference_seconds) - 1),
        button_edges,
        main_stick_changes,
        c_stick_changes,
        _action_summary(submitted),
        _action_summary(expected),
        _action_summary(observed),
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
    player_identity: str | None = None,
    max_frames: int = 28_800,
    rematch: bool = False,
    on_live: Callable[[], None] | None = None,
    stream_id: int = 0,
) -> PlayResult:
    """Play one game after flushing menu inputs from Slippi's delay queue."""
    delay = session.online_delay
    if delay not in runtime.transport_delays:
        raise ValueError(f"session delay {delay} is absent from prepared policy delays {runtime.transport_delays}")
    if max_frames < delay + 3:
        raise ValueError(f"max_frames must be at least transport delay + 3, got {max_frames}")
    first_frame = session.start_rematch(setup) if rematch else session.start_match(setup)
    if session.ego_port is None or session.opponent_port is None:
        raise RuntimeError("netplay ports were not discovered")
    ego_port = session.ego_port
    opponent_port = session.opponent_port
    stage = first_frame.get("stage")
    if not isinstance(stage, int):
        raise RuntimeError(f"first live netplay frame has invalid stage {stage!r}")
    if on_live is not None:
        on_live()
    characters = {port: int(first_frame["ports"][port]["leader"]["post"]["character"]) for port in (1, 2)}
    logger.info(
        "netplay live stream={} delay={} stage={} local_port={} opponent_port={} characters={}:{} imitate={}",
        stream_id,
        delay,
        stage,
        ego_port,
        opponent_port,
        characters[ego_port],
        characters[opponent_port],
        player_identity,
    )
    captured = [first_frame]
    transport = ActionTransport(delay)
    started = time.monotonic()

    # Menu navigation can leave actions in Dolphin's queue. A Slippi time-sync
    # stall can hold one sample, so flush until a neutral state is observed.
    current = first_frame
    for flush_index in range(delay + 120):
        transport.submit(NEUTRAL_CONTROLLER_ACTION)
        current, in_game = session.step(NEUTRAL_CONTROLLER_ACTION)
        captured.append(current)
        if not in_game:
            raise RuntimeError("netplay left live play while flushing menu inputs")
        if flush_index >= delay and controller_actions_match(
            NEUTRAL_CONTROLLER_ACTION,
            _frame_action(current, ego_port),
        ):
            break
    else:
        raise RuntimeError("netplay did not reach a neutral controller state after menu navigation")

    inference_seconds: list[float] = []
    transport_correction_frames = 0
    changed_frames = 0
    button_edges = 0
    main_stick_changes = 0
    c_stick_changes = 0
    previous_submitted: ControllerAction | None = None
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
            stream_id=stream_id,
            frame_id=int(current["id"]),
            controlled_port=ego_port,
            observation=_flat_observation(current, characters),
            applied_action=_frame_action(current, ego_port),
            pending_actions=transport.pending,
            player_identity=player_identity,
            reset=first_policy_frame,
        )
        validate_policy_inputs(policy.spec, runtime, (item,))
        inference_started = time.perf_counter()
        outputs = tuple(policy.step((item,)))
        inference_seconds.append(time.perf_counter() - inference_started)
        action = validate_policy_outputs((item,), outputs)[stream_id]
        if previous_submitted is not None:
            changed_frames += action != previous_submitted
            button_edges += (action.buttons ^ previous_submitted.buttons).bit_count()
            main_stick_changes += (action.main_x, action.main_y) != (
                previous_submitted.main_x,
                previous_submitted.main_y,
            )
            c_stick_changes += (action.c_x, action.c_y) != (
                previous_submitted.c_x,
                previous_submitted.c_y,
            )
        previous_submitted = action
        due = transport.submit(action)
        recent_scheduled.append(due)
        current, in_game = session.step(action)
        captured.append(current)
        if in_game:
            observed = _frame_action(current, ego_port)
            transport_correction_frames += _transport_was_corrected(
                current,
                ego_port,
                due,
                tuple(recent_scheduled),
            )
            if len(inference_seconds) <= 3 or len(inference_seconds) % 600 == 0:
                _log_match_progress(
                    current,
                    ego_port=ego_port,
                    opponent_port=opponent_port,
                    submitted=action,
                    expected=due,
                    observed=observed,
                    inference_seconds=inference_seconds,
                    correction_frames=transport_correction_frames,
                    changed_frames=changed_frames,
                    button_edges=button_edges,
                    main_stick_changes=main_stick_changes,
                    c_stick_changes=c_stick_changes,
                    started=started,
                )
        if not in_game:
            break
        first_policy_frame = False
    else:
        raise RuntimeError(f"netplay game did not finish within {max_frames} frames")

    wall_seconds = time.monotonic() - started
    logger.info(
        "netplay game ended stream={} frames={} wall={:.1f}s policy_p95={:.1f}ms corrections={} "
        "output_changes={}/{} button_edges={} main_changes={} c_changes={}",
        stream_id,
        len(captured),
        wall_seconds,
        _p95_ms(inference_seconds),
        transport_correction_frames,
        changed_frames,
        max(0, len(inference_seconds) - 1),
        button_edges,
        main_stick_changes,
        c_stick_changes,
    )
    return PlayResult(
        trajectory=Trajectory.from_capture(captured, (1, 2)),
        ego_port=ego_port,
        opponent_port=opponent_port,
        stage=stage,
        wall_seconds=wall_seconds,
        inference_seconds=tuple(inference_seconds),
        transport_correction_frames=transport_correction_frames,
    )
