"""Synchronous model play in a local Dolphin session."""

import time
from collections.abc import Mapping

from hal.controller import ControllerAction
from hal.eval import results
from hal.eval.observations import flatten_live_frame
from hal.eval.observations import policy_input_from_frame
from hal.eval.scheduling import ActionScheduler
from hal.eval.scheduling import FrameTiming
from hal.inference.api import PredictionPolicy
from hal.inference.api import PredictionRequest
from hal.inference.api import RuntimeConfig
from hal.inference.transport import ActionTransport
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import controller_actions_match
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.trajectory import Trajectory


def run_local_match(
    session: Session,
    matchup: Matchup,
    policies: Mapping[int, PredictionPolicy],
    runtime: RuntimeConfig,
    timing: FrameTiming,
    *,
    player_identities: Mapping[int, str | None] | None = None,
    desired_return: float | None = 20.0,
    temperature: float = 1.0,
    max_frames: int = 28_800,
    observer: results.PlayObserver | None = None,
) -> results.PlayResult:
    """Play one match, waiting for each model batch before stepping Dolphin.

    The first model-controlled matchup player is the result's ego port. CPU
    players remain under Melee's control and need no policy entry.
    """
    delay = runtime.require_single_delay()
    if timing.input_delay_frames != delay:
        raise ValueError("local timing input delay must match the prepared runtime")
    if max_frames < 2:
        raise ValueError("max_frames must include an observation and one step")
    player_ports = {player.port for player in matchup.players}
    if not policies or not set(policies) <= player_ports:
        raise ValueError("policies must name at least one matchup port and no other ports")
    if len(matchup.players) != 2 or player_ports != {1, 2}:
        raise ValueError("local model play requires matchup ports 1 and 2")
    if any(player.cpu_level > 0 and player.port in policies for player in matchup.players):
        raise ValueError("a CPU port cannot also be model-controlled")
    if player_identities is not None and not set(player_identities) <= set(policies):
        raise ValueError("player identities must name model-controlled ports")

    grouped: dict[int, tuple[PredictionPolicy, list[int]]] = {}
    for port, policy in policies.items():
        if delay not in policy.spec.supported_transport_delays:
            raise ValueError(f"policy on port {port} does not support input delay {delay}")
        group = grouped.setdefault(id(policy), (policy, []))
        group[1].append(port)
    if any(len(ports) > runtime.max_batch_size for _, ports in grouped.values()):
        raise ValueError("one policy receives more controlled ports than its prepared batch size")
    for policy, _ in grouped.values():
        policy.reset_prediction()

    schedulers = {
        port: ActionScheduler(timing, policy.context_frames, generation=1) for port, policy in policies.items()
    }
    transports = {port: ActionTransport(delay) for port in policies}
    matchup_characters = {player.port: int(player.character.value) for player in matchup.players}
    first = session.start_match(matchup)
    stage = first.get("stage")
    if not isinstance(stage, int):
        raise ValueError(f"first local frame has invalid stage {stage!r}")
    captured = [first]
    current = first
    started = time.monotonic()
    last_frame_at = time.perf_counter()
    inference_seconds: list[float] = []
    frame_intervals: list[float] = []
    step_seconds: list[float] = []
    expected_applied: dict[int, ControllerAction] = {}

    while len(captured) < max_frames:
        frame_id = int(current["id"])
        flat = flatten_live_frame(current, matchup_characters)
        requests: dict[int, PredictionRequest] = {}
        for port, policy in policies.items():
            scheduler = schedulers[port]
            item = policy_input_from_frame(
                current,
                spec=policy.spec,
                stream_id=port,
                controlled_port=port,
                pending_actions=transports[port].pending,
                player_identity=None if player_identities is None else player_identities.get(port),
                desired_return=desired_return,
                temperature=temperature,
                reset=not scheduler.history,
                flat=flat,
            )
            expected = expected_applied.get(port)
            if expected is not None and not controller_actions_match(expected, item.applied_action):
                raise RuntimeError(
                    f"local controller alignment failed for port {port} at frame {frame_id}: "
                    f"expected {expected!r}, observed {item.applied_action!r}"
                )
            scheduler.observe(item)
            request = scheduler.request_plan()
            if request is not None:
                requests[port] = request

        for policy, ports in grouped.values():
            batch = tuple(requests[port] for port in ports if port in requests)
            if not batch:
                continue
            inference_started = time.perf_counter()
            plans = tuple(policy.predict(batch))
            elapsed = time.perf_counter() - inference_started
            inference_seconds.append(elapsed)
            if observer is not None:
                observer.observe_policy(elapsed)
            by_stream = {plan.stream_id: plan for plan in plans}
            if len(plans) != len(batch) or set(by_stream) != {request.stream_id for request in batch}:
                raise ValueError("model returned plans for the wrong local streams")
            for port in ports:
                if port in requests:
                    schedulers[port].accept_plan(by_stream[port])

        inputs: dict[int, ControllerInputs] = {}
        for port, scheduler in schedulers.items():
            scheduler.apply_ready_plan(frame_id)
            submitted = scheduler.action_to_submit(frame_id)
            due = transports[port].submit(submitted)
            expected_applied[port] = due
            inputs[port] = due

        step_started = time.perf_counter()
        current, in_game = session.step(inputs)
        frame_at = time.perf_counter()
        next_frame = int(current["id"])
        if next_frame <= frame_id:
            break
        if next_frame != frame_id + 1:
            raise RuntimeError(f"local Dolphin skipped from frame {frame_id} to {next_frame}")
        step_seconds.append(frame_at - step_started)
        frame_intervals.append(frame_at - last_frame_at)
        last_frame_at = frame_at
        captured.append(current)
        if observer is not None:
            observer.observe_frame(int(current["id"]), step_seconds[-1])
        if not in_game:
            break
    else:
        raise RuntimeError(f"local game did not finish within {max_frames} frames")

    ego_port = next(player.port for player in matchup.players if player.port in policies)
    opponent_port = next(player.port for player in matchup.players if player.port != ego_port)
    return results.PlayResult(
        trajectory=Trajectory.from_capture(captured, tuple(player.port for player in matchup.players)),
        ego_port=ego_port,
        opponent_port=opponent_port,
        stage=stage,
        wall_seconds=time.monotonic() - started,
        inference_seconds=tuple(inference_seconds),
        frame_interval_seconds=tuple(frame_intervals),
        dolphin_step_seconds=tuple(step_seconds),
        transport_correction_frames=0,
    )
