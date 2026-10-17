"""Lossy, local overlay telemetry. The producer never waits for a consumer."""

import json
import math
import socket
import time
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import ActionPlan
from hal.inference.api import PredictionRequest
from hal.netplay_service.stream import GameStreamState
from hal.netplay_service.stream import IdleStreamState
from hal.netplay_service.stream import StreamState
from hal.netplay_service.stream import parse_stream_state
from hal.netplay_service.stream import write_stream_state
from hal.netplay_service.value_meter import ValueMeter
from hal.netplay_service.value_meter import ValueSample

MAX_SAMPLE_BYTES = 4096


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    game_id: str
    stream_id: int
    generation: int
    frame_id: int
    sent_at: float
    inputs_at: float
    inputs: ControllerAction
    value: ValueSample | None
    state: GameStreamState
    schema_version: int = 1
    sample_sequence: int = 0


def decode_sample(data: bytes) -> TelemetrySample:
    raw = json.loads(data)
    if (
        not isinstance(raw, dict)
        or set(raw) != set(TelemetrySample.__dataclass_fields__)
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
    ):
        raise ValueError("unsupported telemetry schema")
    if not isinstance(raw["game_id"], str) or not raw["game_id"]:
        raise ValueError("invalid telemetry game")
    for key in ("stream_id", "generation", "frame_id", "sample_sequence"):
        if type(raw[key]) is not int:
            raise ValueError("invalid telemetry identity")
    if raw["generation"] < 1 or raw["sample_sequence"] < 0:
        raise ValueError("invalid telemetry generation or sample sequence")
    state = parse_stream_state(raw["state"])
    if not isinstance(state, GameStreamState) or (state.game_id, state.stream_id, state.generation) != (
        raw["game_id"],
        raw["stream_id"],
        raw["generation"],
    ):
        raise ValueError("telemetry conditioning belongs to another match")
    for key in ("sent_at", "inputs_at"):
        if type(raw[key]) not in (int, float) or not math.isfinite(raw[key]):
            raise ValueError("invalid telemetry timestamp")
    inputs = raw["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != set(ControllerAction.__dataclass_fields__):
        raise ValueError("invalid telemetry inputs")
    for key in ("main_x", "main_y", "c_x", "c_y", "trigger_l", "trigger_r"):
        low = 0 if key.startswith("trigger") else -1
        if type(inputs[key]) not in (int, float) or not low <= inputs[key] <= 1:
            raise ValueError("invalid physical controller input")
    if type(inputs["buttons"]) is not int or not 0 <= inputs["buttons"] <= 65535:
        raise ValueError("invalid physical buttons")
    value = raw["value"]
    if value is not None:
        if not isinstance(value, dict) or set(value) != set(ValueSample.__dataclass_fields__):
            raise ValueError("invalid telemetry value")
        for key in ("stream_id", "generation", "sequence", "source_frame"):
            if type(value[key]) is not int:
                raise ValueError("invalid value identity")
        for key in ("state_value", "ema", "received_at"):
            if type(value[key]) not in (int, float) or not math.isfinite(value[key]):
                raise ValueError("invalid value sample")
        value = ValueSample(**value)
        if (value.stream_id, value.generation) != (raw["stream_id"], raw["generation"]):
            raise ValueError("value belongs to another generation")
    return TelemetrySample(
        raw["game_id"],
        raw["stream_id"],
        raw["generation"],
        raw["frame_id"],
        raw["sent_at"],
        raw["inputs_at"],
        ControllerAction(**inputs),
        value,
        state,
        sample_sequence=raw["sample_sequence"],
    )


class TelemetryPublisher:
    def __init__(self, socket_path: Path, state_path: Path, state: GameStreamState) -> None:
        self.socket_path = socket_path
        self.state_path = state_path
        self.state = state
        self.meter = ValueMeter()
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_NONBLOCK)
        self._frame = -999
        self._inputs = NEUTRAL_CONTROLLER_ACTION
        self._inputs_at = 0.0
        self._sequence = 0
        try:
            write_stream_state(state_path, state)
        except BaseException:
            self._socket.close()
            raise

    def close(self) -> None:
        self._socket.close()
        write_stream_state(self.state_path, IdleStreamState())

    def select_character(self, character: str) -> None:
        self.state = replace(self.state, character=character)

    def observe(self, frame: int, inputs: ControllerAction) -> None:
        self._frame, self._inputs, self._inputs_at = frame, inputs, time.monotonic()
        self._send()

    def requested(self, request: PredictionRequest) -> None:
        if (request.stream_id, request.generation) != (self.state.stream_id, self.state.generation):
            raise ValueError("telemetry request belongs to another match")
        item = request.observations[-1]
        state = replace(
            self.state,
            desired_return=item.desired_return,
            imitation=item.player_identity or "MASKED",
            port=item.controlled_port,
        )
        if state != self.state:
            self.state = state
            self._send()

    def predicted(self, plan: ActionPlan) -> None:
        if (plan.stream_id, plan.generation) != (self.state.stream_id, self.state.generation):
            return
        self.meter.observe(plan)
        self._send()

    def _send(self) -> None:
        self._sequence += 1
        sample = TelemetrySample(
            self.state.game_id,
            self.state.stream_id,
            self.state.generation,
            self._frame,
            time.monotonic(),
            self._inputs_at,
            self._inputs,
            self.meter.snapshot(),
            self.state,
            sample_sequence=self._sequence,
        )
        data = json.dumps(asdict(sample), allow_nan=False, separators=(",", ":")).encode()
        # ENOENT, ECONNREFUSED and queue saturation all mean this frame is dropped.
        with suppress(OSError):
            self._socket.sendto(data, str(self.socket_path))


class TelemetryReceiver:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sample: TelemetrySample | None = None
        self._identity: tuple[str, int, int] | None = None
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_NONBLOCK)
        try:
            path.unlink(missing_ok=True)
            self._socket.bind(str(path))
        except BaseException:
            self._socket.close()
            raise

    def close(self) -> None:
        self._socket.close()
        self.path.unlink(missing_ok=True)

    def state_for(self, state: StreamState) -> StreamState:
        if (
            self.sample is not None
            and isinstance(state, GameStreamState)
            and (state.game_id, state.stream_id, state.generation) == self._identity
        ):
            return self.sample.state
        return state

    def poll(self, state: StreamState, now: float) -> tuple[ControllerAction, float | None]:
        identity = (state.game_id, state.stream_id, state.generation) if isinstance(state, GameStreamState) else None
        if identity != self._identity:
            self.sample = None
            self._identity = identity
        for _ in range(256):
            try:
                data = self._socket.recv(MAX_SAMPLE_BYTES + 1)
            except BlockingIOError:
                break
            if len(data) > MAX_SAMPLE_BYTES:
                continue
            try:
                sample = decode_sample(data)
            except ValueError, TypeError, KeyError:
                continue
            if (sample.game_id, sample.stream_id, sample.generation) != identity:
                continue
            if not 0 <= now - sample.sent_at <= 1:
                continue
            if self.sample is not None and (
                sample.frame_id < self.sample.frame_id or sample.sample_sequence <= self.sample.sample_sequence
            ):
                continue
            self.sample = sample
        sample = self.sample
        if sample is None:
            return NEUTRAL_CONTROLLER_ACTION, None
        inputs = sample.inputs if 0 <= now - sample.inputs_at <= 0.25 else NEUTRAL_CONTROLLER_ACTION
        value = sample.value
        shown = (
            value.ema if value is not None and value.source_frame >= 0 and 0 <= now - value.received_at <= 1 else None
        )
        return inputs, shown
