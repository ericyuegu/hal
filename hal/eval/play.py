"""Direct netplay loop for the public policy interface."""

import math
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import peppi_py
from loguru import logger
from peppi_py.game import EndMethod

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import POLICY_BUTTON_MASK
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
    frame_interval_seconds: tuple[float, ...]
    dolphin_step_seconds: tuple[float, ...]
    transport_correction_frames: int

    @property
    def inference_p95_ms(self) -> float:
        """Nearest-rank p95 of the complete policy call."""
        return _p95_ms(self.inference_seconds)

    @property
    def game_fps(self) -> float:
        elapsed = sum(self.frame_interval_seconds)
        return len(self.frame_interval_seconds) / elapsed if elapsed > 0 else 0.0

    @property
    def frame_interval_p95_ms(self) -> float:
        return _p95_ms(self.frame_interval_seconds)

    @property
    def dolphin_step_p95_ms(self) -> float:
        return _p95_ms(self.dolphin_step_seconds)


@dataclass(frozen=True, slots=True)
class ReplayEnd:
    path: Path
    method: EndMethod

    @property
    def completed(self) -> bool:
        return self.method in (EndMethod.TIME, EndMethod.GAME, EndMethod.RESOLVED)


class PlayObserver(Protocol):
    def observe_policy(self, seconds: float) -> None: ...

    def observe_frame(self, frame_id: int, dolphin_step_seconds: float) -> None: ...


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


def _policy_applied_action(frame: dict, port: int) -> ControllerAction:
    action = _frame_action(frame, port)
    return ControllerAction(
        main_x=action.main_x,
        main_y=action.main_y,
        c_x=action.c_x,
        c_y=action.c_y,
        trigger_l=action.trigger_l,
        trigger_r=action.trigger_r,
        buttons=action.buttons & POLICY_BUTTON_MASK,
    )


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


def read_new_replay_end(replay_dir: Path, previous: Collection[Path]) -> ReplayEnd:
    """Read the game-end record from this invocation's replay."""
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
    return ReplayEnd(replay, method)


def require_completed_replay(replay_dir: Path, previous: Collection[Path]) -> Path:
    """Return this invocation's replay after validating its game-end record."""
    replay_end = read_new_replay_end(replay_dir, previous)
    if not replay_end.completed:
        raise RuntimeError(f"netplay replay ended via {replay_end.method.name}: {replay_end.path}")
    return replay_end.path


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
    observer: PlayObserver | None = None,
    stream_id: int = 0,
) -> PlayResult:
    """Play one game from an exact frame-zero session boundary."""
    delay = session.online_delay
    if delay not in runtime.transport_delays:
        raise ValueError(f"session delay {delay} is absent from prepared policy delays {runtime.transport_delays}")
    if max_frames < delay + 3:
        raise ValueError(f"max_frames must be at least transport delay + 3, got {max_frames}")
    transport = ActionTransport(delay)
    # Slippi 3.6.4 can replay a recent local pad during its 30-frame clock
    # correction. Its rollback window is seven frames, so retain the current
    # scheduled action and the seven that precede it.
    recent_scheduled: deque[ControllerAction] = deque(
        (NEUTRAL_CONTROLLER_ACTION,),
        maxlen=8,
    )
    countdown_policy_frames = 0
    expected_countdown_action: ControllerAction | None = None
    transport_correction_frames = 0

    def countdown_action(frame: dict) -> ControllerAction:
        nonlocal countdown_policy_frames
        nonlocal expected_countdown_action
        nonlocal transport_correction_frames
        if session.ego_port is None:
            raise RuntimeError("netplay local port was not discovered before the countdown")
        ego_port = session.ego_port
        if expected_countdown_action is not None and countdown_policy_frames > delay:
            transport_correction_frames += _transport_was_corrected(
                frame,
                ego_port,
                expected_countdown_action,
                tuple(recent_scheduled),
            )
        characters = {port: int(frame["ports"][port]["leader"]["post"]["character"]) for port in (1, 2)}
        item = PolicyInput(
            stream_id=stream_id,
            frame_id=int(frame["id"]),
            controlled_port=ego_port,
            observation=_flat_observation(frame, characters),
            applied_action=_policy_applied_action(frame, ego_port),
            pending_actions=transport.pending,
            player_identity=player_identity,
            reset=countdown_policy_frames == 0,
        )
        validate_policy_inputs(policy.spec, runtime, (item,))
        action = validate_policy_outputs((item,), tuple(policy.step((item,))))[stream_id]
        expected_countdown_action = transport.submit(action)
        recent_scheduled.append(expected_countdown_action)
        countdown_policy_frames += 1
        return action

    if rematch:
        first_frame = session.start_rematch(setup, on_countdown_frame=countdown_action)
    else:
        first_frame = session.start_match(setup, on_countdown_frame=countdown_action)
    if session.ego_port is None or session.opponent_port is None:
        raise RuntimeError("netplay ports were not discovered")
    ego_port = session.ego_port
    opponent_port = session.opponent_port
    if first_frame.get("id") != 0:
        raise RuntimeError(f"netplay session started at frame {first_frame.get('id')!r}; expected frame 0")
    if expected_countdown_action is not None and countdown_policy_frames > delay:
        transport_correction_frames += _transport_was_corrected(
            first_frame,
            ego_port,
            expected_countdown_action,
            tuple(recent_scheduled),
        )
    stage = first_frame.get("stage")
    if not isinstance(stage, int):
        raise RuntimeError(f"first live netplay frame has invalid stage {stage!r}")
    if on_live is not None:
        on_live()
    characters = {port: int(first_frame["ports"][port]["leader"]["post"]["character"]) for port in (1, 2)}
    logger.info(
        "netplay live stream={} delay={} stage={} local_port={} opponent_port={} characters={}:{} "
        "imitate={} countdown_frames={}",
        stream_id,
        delay,
        stage,
        ego_port,
        opponent_port,
        characters[ego_port],
        characters[opponent_port],
        player_identity,
        countdown_policy_frames,
    )
    captured = [first_frame]
    started = time.monotonic()
    current = first_frame

    inference_seconds: list[float] = []
    changed_frames = 0
    button_edges = 0
    main_stick_changes = 0
    c_stick_changes = 0
    previous_submitted: ControllerAction | None = None
    first_policy_frame = countdown_policy_frames == 0
    frame_interval_seconds: list[float] = []
    dolphin_step_seconds: list[float] = []
    last_frame_at = time.perf_counter()
    while len(captured) < max_frames:
        item = PolicyInput(
            stream_id=stream_id,
            frame_id=int(current["id"]),
            controlled_port=ego_port,
            observation=_flat_observation(current, characters),
            applied_action=_policy_applied_action(current, ego_port),
            pending_actions=transport.pending,
            player_identity=player_identity,
            reset=first_policy_frame,
        )
        validate_policy_inputs(policy.spec, runtime, (item,))
        inference_started = time.perf_counter()
        outputs = tuple(policy.step((item,)))
        inference_elapsed = time.perf_counter() - inference_started
        inference_seconds.append(inference_elapsed)
        if observer is not None:
            observer.observe_policy(inference_elapsed)
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
        dolphin_step_started = time.perf_counter()
        current, in_game = session.step(action)
        frame_at = time.perf_counter()
        dolphin_step_seconds.append(frame_at - dolphin_step_started)
        frame_interval_seconds.append(frame_at - last_frame_at)
        last_frame_at = frame_at
        captured.append(current)
        if in_game:
            if observer is not None:
                observer.observe_frame(int(current["id"]), dolphin_step_seconds[-1])
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
    frame_interval_values = tuple(frame_interval_seconds)
    dolphin_step_values = tuple(dolphin_step_seconds)
    game_fps = len(frame_interval_values) / sum(frame_interval_values)
    logger.info(
        "netplay game ended stream={} frames={} wall={:.1f}s fps={:.1f} frame_p95={:.1f}ms "
        "dolphin_p95={:.1f}ms policy_p95={:.1f}ms corrections={} output_changes={}/{} "
        "button_edges={} main_changes={} c_changes={}",
        stream_id,
        len(captured),
        wall_seconds,
        game_fps,
        _p95_ms(frame_interval_values),
        _p95_ms(dolphin_step_values),
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
        frame_interval_seconds=frame_interval_values,
        dolphin_step_seconds=dolphin_step_values,
        transport_correction_frames=transport_correction_frames,
    )
