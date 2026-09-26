"""Nonblocking netplay with inference running beside Dolphin."""

import time
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

from loguru import logger

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval import results
from hal.eval.observations import policy_input_from_frame
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import RuntimeConfig
from hal.inference.worker import InferenceClient
from hal.inference.worker import InferenceUnavailable
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.session import FrameTimeout
from hal.sim.trajectory import Trajectory


class ScheduleObserver(Protocol):
    def observe_schedule(self, schedule: ActionScheduler) -> None: ...


class DolphinConnectionLost(RuntimeError):
    """The live emulator stopped delivering controller or frame data."""


def run_netplay_match(
    session: NetplaySession,
    setup: NetplaySetup,
    client: InferenceClient,
    runtime: RuntimeConfig,
    timing: FrameTiming,
    *,
    player_identity: str | None = None,
    policy_settings: Callable[[], tuple[float | None, float]] | None = None,
    max_frames: int = 28_800,
    rematch: bool = False,
    on_live: Callable[[], None] | None = None,
    observer: results.PlayObserver | None = None,
    schedule_observer: ScheduleObserver | None = None,
    stream_id: int = 0,
) -> results.PlayResult:
    if session.online_delay not in runtime.transport_delays or not session.realtime:
        raise ValueError("netplay requires a prepared input delay and a nonblocking session")
    if timing.input_delay_frames != session.online_delay:
        raise ValueError("netplay timing input delay differs from the session")
    schedule = ActionScheduler(timing, client.context_frames, client.start_match())
    inference_seconds: list[float] = []
    matchup_characters: dict[int, int] | None = None

    def advance(frame: dict) -> None:
        nonlocal matchup_characters
        if session.ego_port is None:
            raise RuntimeError("local port was not discovered before countdown")
        if matchup_characters is None:
            matchup_characters = {port: int(frame["ports"][port]["leader"]["post"]["character"]) for port in (1, 2)}
        port = session.ego_port
        desired_return, temperature = (20.0, 1.0) if policy_settings is None else policy_settings()
        frame_id = int(frame["id"])
        if schedule.history and frame_id > schedule.history[-1].frame_id + 1:
            raise RuntimeError(
                f"netplay skipped observation frame {schedule.history[-1].frame_id + 1} before {frame_id}"
            )
        pending = tuple(
            schedule.submitted.get(f, NEUTRAL_CONTROLLER_ACTION)
            for f in range(frame_id + 1, frame_id + session.online_delay + 1)
        )
        item = policy_input_from_frame(
            frame,
            spec=client.spec,
            stream_id=stream_id,
            controlled_port=port,
            pending_actions=pending,
            player_identity=player_identity,
            desired_return=desired_return,
            temperature=temperature,
            reset=not schedule.history,
            matchup_characters=matchup_characters,
        )
        schedule.observe(item)

    def choose(frame_id: int) -> ControllerAction:
        if not schedule.inference_failed:
            try:
                response = client.poll()
                if response is not None and schedule.accept_plan(response):
                    inference_seconds.append(client.last_latency)
                    if observer is not None and frame_id >= 0:
                        observer.observe_policy(client.last_latency)
                schedule.apply_ready_plan(frame_id)
                if not client.busy:
                    request = schedule.request_plan()
                    if request is not None:
                        client.submit(request)
            except InferenceUnavailable as error:
                logger.error("netplay inference lost; draining current plan: {}", error)
                schedule.fail_inference()
        action = schedule.action_to_submit(frame_id)
        if schedule_observer is not None and frame_id >= 0:
            schedule_observer.observe_schedule(schedule)
        if schedule.drained(frame_id):
            with suppress(OSError, EOFError):
                session.submit(NEUTRAL_CONTROLLER_ACTION)
            raise InferenceUnavailable("service failure: inference lost; bot forfeits after draining its plan")
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
            raise DolphinConnectionLost("Dolphin connection lost during netplay") from error
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
    return results.PlayResult(
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
