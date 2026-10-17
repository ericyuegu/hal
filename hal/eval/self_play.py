"""Two real-time inference streams driving one cancellable local session."""

import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass

from hal.eval.observations import policy_input_from_frame
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.client import InferenceClient
from hal.inference.client import InferenceUnavailable
from hal.netplay_service.telemetry import TelemetryPublisher
from hal.sim.inputs import ActionTransport
from hal.sim.inputs import canonical_pre_to_action
from hal.sim.session import Matchup
from hal.sim.session import Session

TIMING = FrameTiming(2, 2, 4, 4, 8)


@dataclass(frozen=True, slots=True)
class SelfPlayMetrics:
    frames: int
    wall_seconds: float
    inference_seconds: tuple[float, ...]


def run_match(
    session: Session,
    matchup: Matchup,
    clients: tuple[InferenceClient, InferenceClient],
    stream_ids: tuple[int, int],
    imitations: tuple[str, str],
    telemetry: TelemetryPublisher,
    observe_frame: Callable[[int, float], None],
    on_live: Callable[[], None],
    *,
    max_frames: int,
    observe_policy: Callable[[float], None] | None = None,
) -> SelfPlayMetrics:
    schedules: list[ActionScheduler] = []
    transports = (ActionTransport(2), ActionTransport(2))
    latencies: list[float] = []
    with ExitStack() as streams:
        for client, stream_id in zip(clients, stream_ids, strict=True):
            generation = client.start_match(stream_id, TIMING.fixed_prefix_frames)
            streams.callback(client.close_match)
            schedules.append(ActionScheduler(TIMING, client.context_frames, generation))
        frame = session.start_match(matchup)
        characters = {player.port: int(player.character.value) for player in matchup.players}
        first = int(frame["id"])
        started = time.monotonic()
        on_live()
        for _ in range(max_frames):
            inputs = {}
            frame_id = int(frame["id"])
            for index, client in enumerate(clients):
                port = index + 1
                schedule = schedules[index]
                item = policy_input_from_frame(
                    frame,
                    spec=client.spec,
                    stream_id=stream_ids[index],
                    controlled_port=port,
                    player_identity=None if imitations[index] == "MASKED" else imitations[index],
                    desired_return=120.0,
                    temperature=1.0,
                    reset=not schedule.history,
                    matchup_characters=characters,
                )
                schedule.observe(item)
                if index == 0:
                    telemetry.observe(frame_id, canonical_pre_to_action(frame["ports"][port]["leader"]["pre"]))
                response = client.poll()
                if response is not None:
                    schedule.accept_plan(response, frame_id)
                    latencies.append(client.last_latency)
                    if observe_policy is not None:
                        observe_policy(client.last_latency)
                    if index == 0:
                        telemetry.predicted(response)
                if not client.busy:
                    request = schedule.request_plan()
                    if request is not None:
                        client.submit(request)
                        if index == 0:
                            telemetry.requested(request)
                inputs[port] = transports[index].submit(schedule.action_to_submit(frame_id))
                if frame_id > 120 and schedule.neutral_fallback_frames > 120:
                    raise InferenceUnavailable("self-play exhausted its action plan")
            step_started = time.monotonic()
            frame, playing = session.step(inputs)
            observe_frame(frame_id, time.monotonic() - step_started)
            if not playing:
                return SelfPlayMetrics(frame_id - first + 1, time.monotonic() - started, tuple(latencies))
        raise RuntimeError("self-play exceeded its frame limit")
