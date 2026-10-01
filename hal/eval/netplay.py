"""Nonblocking netplay with inference running beside Dolphin."""

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from typing import Final
from typing import Protocol

from loguru import logger

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.eval import results
from hal.eval.observations import policy_input_from_frame
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.eval.scheduling import PlanDecision
from hal.eval.scheduling import PlanProtocolError
from hal.inference.api import ActionPlan
from hal.inference.api import RuntimeConfig
from hal.inference.client import InferenceClient
from hal.inference.client import InferenceUnavailable
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup
from hal.sim.session import FrameTimeout
from hal.sim.trajectory import Trajectory


class ScheduleObserver(Protocol):
    def observe_schedule(self, schedule: ActionScheduler) -> None: ...


class DolphinConnectionLost(RuntimeError):
    """The live emulator stopped delivering controller or frame data."""


class NoUsableActionPlan(RuntimeError):
    """Valid inference replies could not provide actions before the match deadline."""


# A paused netplay game emits no frames; this much silence counts as a pause.
_PAUSE_DETECT_SECONDS: Final[float] = 2.0


class PausedTooLong(RuntimeError):
    """The game stayed paused past the allowed pause."""


class _NetplayLifecycle:
    """Hold the per-match observation, request, and exhaustion state."""

    def __init__(
        self,
        session: NetplaySession,
        client: InferenceClient,
        schedule: ActionScheduler,
        *,
        player_identity: Callable[[], str | None] | str | None,
        policy_settings: Callable[[], tuple[float | None, float]] | None,
        observer: results.PlayObserver | None,
        schedule_observer: ScheduleObserver | None,
        stream_id: int,
        pause_seconds: float | None = None,
        on_pause: Callable[[bool], None] | None = None,
        on_prediction: Callable[[ActionPlan], None] | None = None,
    ) -> None:
        self.session = session
        self.client = client
        self.schedule = schedule
        self._player_identity = player_identity
        self._resolved_player_identity: str | None = None
        self._identity_resolved = False
        self.policy_settings = policy_settings
        self.observer = observer
        self.schedule_observer = schedule_observer
        self.stream_id = stream_id
        self.pause_seconds = pause_seconds
        self.on_pause = on_pause
        self.on_prediction = on_prediction
        self.inference_seconds: list[float] = []
        self.inference_source_frames: list[int] = []
        self.plan_decisions: list[PlanDecision] = []
        self.observed_frame_ids: list[int] = []
        self.frame_interval_seconds: list[float] = []
        self.dolphin_step_seconds: list[float] = []
        self.schedule_events: list[results.ScheduleEvent] = []
        self._last_schedule_counts: tuple[int | bool, ...] | None = None
        self.matchup_characters: dict[int, int] | None = None
        self.exhausted_since: float | None = None

    def advance(self, frame: dict) -> None:
        if self.session.ego_port is None:
            raise RuntimeError("local port was not discovered before countdown")
        if self.matchup_characters is None:
            self.matchup_characters = {
                port: int(frame["ports"][port]["leader"]["post"]["character"]) for port in (1, 2)
            }
        desired_return, temperature = (20.0, 1.0) if self.policy_settings is None else self.policy_settings()
        frame_id = int(frame["id"])
        if self.schedule.history and frame_id > self.schedule.history[-1].frame_id + 1:
            raise RuntimeError(
                f"netplay skipped observation frame {self.schedule.history[-1].frame_id + 1} before {frame_id}"
            )
        if not self._identity_resolved:
            if self._player_identity is None or isinstance(self._player_identity, str):
                self._resolved_player_identity = self._player_identity
            else:
                self._resolved_player_identity = self._player_identity()
            self._identity_resolved = True
        item = policy_input_from_frame(
            frame,
            spec=self.client.spec,
            stream_id=self.stream_id,
            controlled_port=self.session.ego_port,
            player_identity=self._resolved_player_identity,
            desired_return=desired_return,
            temperature=temperature,
            reset=not self.schedule.history,
            matchup_characters=self.matchup_characters,
        )
        if self.schedule.observe(item):
            self.observed_frame_ids.append(frame_id)

    def choose(self, frame_id: int) -> ControllerAction:
        if not self.schedule.inference_failed:
            try:
                response = self.client.poll()
                if response is not None:
                    self.schedule.accept_plan(response, frame_id)
                    decision = self.schedule.last_decision
                    if decision is None:
                        raise PlanProtocolError("inference response had no active request")
                    if self.on_prediction is not None:
                        self.on_prediction(response)
                    self.plan_decisions.append(decision)
                    self.inference_seconds.append(self.client.last_latency)
                    self.inference_source_frames.append(response.source_frame)
                    if self.observer is not None and frame_id >= 0:
                        self.observer.observe_policy(self.client.last_latency)
                if not self.client.busy:
                    request = self.schedule.request_plan()
                    if request is not None:
                        self.client.submit(request)
            except (InferenceUnavailable, PlanProtocolError) as error:
                logger.error("netplay inference lost; draining current plan: {}", error)
                self.schedule.fail_inference()
        action = self.schedule.action_to_submit(frame_id)
        self._record_schedule_change(frame_id)
        # Countdown may use neutral actions while the first plan is prepared.
        # Once gameplay has consumed a valid response, repeated unusable plans
        # must not leave a match on neutral input indefinitely.
        if (
            frame_id >= 0
            and self.schedule.last_submission_used_fallback
            and (self.schedule.exhausted_chunks or self.schedule.last_consumed_source_frame is not None)
        ):
            if self.exhausted_since is None:
                self.exhausted_since = time.monotonic()
            elif time.monotonic() - self.exhausted_since >= 2.0:
                raise NoUsableActionPlan("service failure: no usable action plan for two seconds")
        else:
            self.exhausted_since = None
        if self.schedule_observer is not None and frame_id >= 0:
            self.schedule_observer.observe_schedule(self.schedule)
        if self.schedule.drained(frame_id):
            with suppress(OSError, EOFError):
                self.session.submit(NEUTRAL_CONTROLLER_ACTION)
            raise InferenceUnavailable("service failure: inference lost; bot forfeits after draining its plan")
        return action

    def choose_countdown(self, frame: dict) -> ControllerAction:
        return self.choose(int(frame["id"]))

    def _record_schedule_change(self, frame_id: int) -> None:
        schedule = self.schedule
        counts: tuple[int | bool, ...] = (
            schedule.deadline_misses,
            schedule.prefix_mismatches,
            schedule.exhausted_chunks,
            schedule.neutral_fallback_frames,
            schedule.submission_gaps,
            schedule.transport_corrections,
            schedule.inference_failed,
        )

        if counts == self._last_schedule_counts:
            return
        self._last_schedule_counts = counts
        self.schedule_events.append(
            results.ScheduleEvent(
                choice_frame=frame_id,
                target_frame=frame_id + schedule.timing.physical_delay_frames + 1,
                phase="countdown" if frame_id < 0 else "gameplay",
                deadline_misses=schedule.deadline_misses,
                prefix_mismatches=schedule.prefix_mismatches,
                exhausted_chunks=schedule.exhausted_chunks,
                neutral_fallback_frames=schedule.neutral_fallback_frames,
                submission_gaps=schedule.submission_gaps,
                transport_corrections=schedule.transport_corrections,
                inference_failed=schedule.inference_failed,
            )
        )

    def progress(self) -> results.NetplayProgress:
        return results.NetplayProgress(
            stream_id=self.stream_id,
            generation=self.schedule.generation,
            pending_sequence=None if self.schedule.request is None else self.schedule.request.sequence,
            last_consumed_source_frame=self.schedule.last_consumed_source_frame,
            last_accepted_source_frame=self.schedule.last_accepted_source_frame,
            observed_frame_ids=tuple(self.observed_frame_ids),
            inference_source_frames=tuple(self.inference_source_frames),
            inference_seconds=tuple(self.inference_seconds),
            frame_interval_seconds=tuple(self.frame_interval_seconds),
            dolphin_step_seconds=tuple(self.dolphin_step_seconds),
            schedule_events=tuple(self.schedule_events),
            plan_decisions=tuple(self.plan_decisions),
        )

    def run(
        self,
        setup: NetplaySetup,
        *,
        max_frames: int,
        rematch: bool,
        on_live: Callable[[], None] | None,
    ) -> results.PlayResult:
        connection_started = time.monotonic()
        first = (self.session.start_rematch if rematch else self.session.start_match)(
            setup, on_countdown_frame=self.choose_countdown, on_countdown_observation=self.advance
        )
        if self.session.ego_port is None or self.session.opponent_port is None or first.get("id") != 0:
            raise RuntimeError("netplay did not establish frame zero and both ports")
        if on_live is not None:
            on_live()
        connection_countdown_seconds = time.monotonic() - connection_started
        started = time.monotonic()
        previous_at = time.perf_counter()
        captured = [first]
        self.advance(first)
        current = first
        stalled_since: float | None = None
        while len(captured) < max_frames:
            try:
                if stalled_since is None:
                    self.session.submit(self.choose(int(current["id"])))
                step_started = time.perf_counter()
                if self.pause_seconds is None:
                    frames, in_game = self.session.read_frames()
                else:
                    frames, in_game = self.session.read_frames(timeout_seconds=_PAUSE_DETECT_SECONDS)
            except FrameTimeout as error:
                if self.pause_seconds is None:
                    with suppress(OSError, EOFError):
                        self.session.submit(NEUTRAL_CONTROLLER_ACTION)
                    raise DolphinConnectionLost("Dolphin connection lost during netplay") from error
                now = time.monotonic()
                if stalled_since is None:
                    stalled_since = now - _PAUSE_DETECT_SECONDS
                    if self.on_pause is not None:
                        self.on_pause(True)
                if now - stalled_since >= self.pause_seconds:
                    raise PausedTooLong(f"no frames for {now - stalled_since:.0f}s") from error
                continue
            except (OSError, EOFError) as error:
                with suppress(OSError, EOFError):
                    self.session.submit(NEUTRAL_CONTROLLER_ACTION)
                raise DolphinConnectionLost("Dolphin connection lost during netplay") from error
            if stalled_since is not None:
                stalled_since = None
                if self.on_pause is not None:
                    self.on_pause(False)
            self.dolphin_step_seconds.append(time.perf_counter() - step_started)
            # Every received observation is retained, but only the latest can select an input.
            for frame in frames if in_game else frames[:-1]:
                self.advance(frame)
            for received_at in self.session.frame_times:
                self.frame_interval_seconds.append(received_at - previous_at)
                previous_at = received_at
            captured.extend(frames)
            current = frames[-1]
            if not in_game:
                break
            if self.observer is not None:
                self.observer.observe_frame(int(current["id"]), self.dolphin_step_seconds[-1])
        else:
            raise RuntimeError(f"netplay game did not finish within {max_frames} frames")
        game_ended = time.monotonic()
        logger.info(
            "real-time netplay ended stream={} generation={} schedule={} deadline_misses={} "
            "prefix_mismatches={} exhausted_chunks={} neutral_fallback_frames={} transport_corrections={}",
            self.stream_id,
            self.schedule.generation,
            self.schedule.timing,
            self.schedule.deadline_misses,
            self.schedule.prefix_mismatches,
            self.schedule.exhausted_chunks,
            self.schedule.neutral_fallback_frames,
            self.schedule.transport_corrections,
        )
        result = results.PlayResult(
            Trajectory.from_capture(captured, (1, 2)),
            self.session.ego_port,
            self.session.opponent_port,
            int(first["stage"]),
            game_ended - started,
            tuple(self.inference_seconds),
            tuple(self.frame_interval_seconds),
            tuple(self.dolphin_step_seconds),
            self.schedule.transport_corrections,
            tuple(self.inference_source_frames),
            tuple(self.schedule_events),
            connection_countdown_seconds,
            plan_decisions=tuple(self.plan_decisions),
            generation=self.schedule.generation,
        )
        return replace(result, match_end_seconds=time.monotonic() - game_ended)


def run_netplay_match(
    session: NetplaySession,
    setup: NetplaySetup,
    client: InferenceClient,
    runtime: RuntimeConfig,
    timing: FrameTiming,
    *,
    player_identity: Callable[[], str | None] | str | None = None,
    policy_settings: Callable[[], tuple[float | None, float]] | None = None,
    max_frames: int = 28_800,
    rematch: bool = False,
    on_live: Callable[[], None] | None = None,
    on_prediction: Callable[[ActionPlan], None] | None = None,
    on_failure: Callable[[results.NetplayProgress, BaseException], None] | None = None,
    pause_seconds: float | None = None,
    on_pause: Callable[[bool], None] | None = None,
    observer: results.PlayObserver | None = None,
    schedule_observer: ScheduleObserver | None = None,
    stream_id: int = 0,
) -> results.PlayResult:
    if session.online_delay not in runtime.transport_delays or not session.realtime:
        raise ValueError("netplay requires a prepared input delay and a nonblocking session")
    if timing.physical_delay_frames != session.online_delay:
        raise ValueError("netplay timing input delay differs from the session")
    generation = client.start_match(stream_id, timing.fixed_prefix_frames)
    lifecycle: _NetplayLifecycle | None = None
    try:
        schedule = ActionScheduler(timing, client.context_frames, generation)
        lifecycle = _NetplayLifecycle(
            session,
            client,
            schedule,
            player_identity=player_identity,
            policy_settings=policy_settings,
            observer=observer,
            schedule_observer=schedule_observer,
            stream_id=stream_id,
            pause_seconds=pause_seconds,
            on_pause=on_pause,
            on_prediction=on_prediction,
        )
        result = lifecycle.run(setup, max_frames=max_frames, rematch=rematch, on_live=on_live)
    except BaseException as error:
        if on_failure is not None and lifecycle is not None:
            try:
                on_failure(lifecycle.progress(), error)
            except Exception as capture_error:
                error.add_note(f"netplay failure capture failed: {type(capture_error).__name__}: {capture_error}")
        if isinstance(error, NoUsableActionPlan):
            # This failure is local to the match only if the stream can still be
            # released. A failed release means the inference connection is lost.
            client.close_match()
        else:
            with suppress(InferenceUnavailable):
                client.close_match()
        raise
    else:
        release_started = time.monotonic()
        client.close_match()
        return replace(result, match_end_seconds=result.match_end_seconds + time.monotonic() - release_started)
