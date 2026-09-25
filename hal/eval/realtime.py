"""Real-time netplay: the Dolphin worker runs independently of model inference."""

import time
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

from loguru import logger

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import POLICY_BUTTON_MASK
from hal.controller import ControllerAction
from hal.eval.play import PlayObserver
from hal.eval.play import PlayResult
from hal.inference.api import PolicyInput
from hal.inference.api import RuntimeConfig
from hal.netplay_service.chunks import EngineLost
from hal.netplay_service.chunks import RemoteChunkPolicy
from hal.netplay_service.schedule import FrameSchedule
from hal.sim.inputs import canonical_pre_to_action
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.session import FrameTimeout
from hal.sim.trajectory import Trajectory
from hal.training.canonical import flatten_canonical_frame


class ScheduleObserver(Protocol):
    def observe_schedule(self, schedule: FrameSchedule) -> None: ...


def run_realtime_match(
    session: NetplaySession,
    setup: NetplaySetup,
    policy: RemoteChunkPolicy,
    runtime: RuntimeConfig,
    *,
    player_identity: str | None = None,
    policy_settings: Callable[[], tuple[float | None, float]] | None = None,
    max_frames: int = 28_800,
    rematch: bool = False,
    on_live: Callable[[], None] | None = None,
    observer: PlayObserver | None = None,
    schedule_observer: ScheduleObserver | None = None,
    stream_id: int = 0,
) -> PlayResult:
    if session.online_delay not in runtime.transport_delays or not session.realtime:
        raise ValueError("real-time match requires a calibrated transport delay and a nonblocking session")
    timing = next((s for s in policy.schedules if s.transport_frames == session.online_delay), None)
    if timing is None:
        raise ValueError("session transport delay was not calibrated")
    schedule = FrameSchedule(timing, policy.context_frames, policy.start_match())
    inference_seconds: list[float] = []

    def advance(frame: dict) -> ControllerAction:
        if session.ego_port is None:
            raise RuntimeError("local port was not discovered before countdown")
        port = session.ego_port
        action = canonical_pre_to_action(frame["ports"][port]["leader"]["pre"])
        applied = ControllerAction(
            action.main_x,
            action.main_y,
            action.c_x,
            action.c_y,
            action.trigger_l,
            action.trigger_r,
            action.buttons & POLICY_BUTTON_MASK,
        )
        characters = {p: int(frame["ports"][p]["leader"]["post"]["character"]) for p in (1, 2)}
        flat = flatten_canonical_frame({**frame, "_matchup": {"stage": frame["stage"], "character": characters}})
        desired_return, temperature = (20.0, 1.0) if policy_settings is None else policy_settings()
        frame_id = int(frame["id"])
        pending = tuple(
            schedule.submitted.get(f, NEUTRAL_CONTROLLER_ACTION)
            for f in range(frame_id + 1, frame_id + session.online_delay + 1)
        )
        item = PolicyInput(
            stream_id,
            frame_id,
            port,
            flat,
            applied,
            pending,
            player_identity,
            desired_return,
            temperature,
            reset=not schedule.history,
        )
        schedule.observe(item)
        return applied

    def choose(frame_id: int) -> ControllerAction:
        if not schedule.engine_failed:
            try:
                response = policy.poll()
                if response is not None and schedule.receive(response):
                    inference_seconds.append(policy.last_latency)
                    if observer is not None and frame_id >= 0:
                        observer.observe_policy(policy.last_latency)
                schedule.handoff(frame_id)
                if not policy.busy:
                    request = schedule.begin_request()
                    if request is not None:
                        policy.submit(request)
            except EngineLost as error:
                logger.error("netplay inference lost; draining current plan: {}", error)
                schedule.fail_engine()
        action = schedule.submit(frame_id)
        if schedule_observer is not None and frame_id >= 0:
            schedule_observer.observe_schedule(schedule)
        if schedule.drained(frame_id):
            session.submit(NEUTRAL_CONTROLLER_ACTION)
            raise EngineLost("service failure: inference lost; bot forfeits after draining its plan")
        return action

    def countdown(frame: dict) -> ControllerAction:
        return choose(int(frame["id"]))

    def observe_countdown(frame: dict) -> None:
        advance(frame)

    first = (session.start_rematch if rematch else session.start_match)(
        setup, on_countdown_frame=countdown, on_countdown_observation=observe_countdown
    )
    if session.ego_port is None or session.opponent_port is None or first.get("id") != 0:
        raise RuntimeError("netplay did not establish frame zero and both ports")
    if on_live is not None:
        on_live()
    started = time.monotonic()
    previous_at = time.perf_counter()
    captured = [first]
    advance(first)
    intervals: list[float] = []
    steps: list[float] = []
    current = first
    while len(captured) < max_frames:
        try:
            session.submit(choose(int(current["id"])))
            step_started = time.perf_counter()
            frames, in_game = session.read_frames()
        except (OSError, EOFError, FrameTimeout) as error:
            with suppress(OSError, EOFError):
                session.submit(NEUTRAL_CONTROLLER_ACTION)
            raise EngineLost("service failure: Dolphin connection lost; bot forfeits") from error
        steps.append(time.perf_counter() - step_started)
        # Every received observation is retained, but only the latest can select an input.
        for frame in frames if in_game else frames[:-1]:
            advance(frame)
        for received_at in session.frame_times:
            intervals.append(received_at - previous_at)
            previous_at = received_at
        captured.extend(frames)
        current = frames[-1]
        if not in_game:
            break
        if observer is not None:
            observer.observe_frame(int(current["id"]), steps[-1])
    else:
        raise RuntimeError(f"netplay game did not finish within {max_frames} frames")
    logger.info(
        "real-time netplay ended stream={} generation={} schedule={} deadline_misses={} "
        "prefix_mismatches={} exhausted_chunks={} neutral_fallback_frames={} transport_corrections={}",
        stream_id,
        schedule.generation,
        timing,
        schedule.deadline_misses,
        schedule.prefix_mismatches,
        schedule.exhausted_chunks,
        schedule.neutral_fallback_frames,
        schedule.transport_corrections,
    )
    return PlayResult(
        Trajectory.from_capture(captured, (1, 2)),
        session.ego_port,
        session.opponent_port,
        int(first["stage"]),
        time.monotonic() - started,
        tuple(inference_seconds),
        tuple(intervals),
        tuple(steps),
        schedule.transport_corrections,
    )
