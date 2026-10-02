"""One Cloudflare connection shared by all local runner processes."""

import json
import random
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import cast

import httpx
from loguru import logger
from websockets.exceptions import WebSocketException
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect

from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import read_slot_status
from hal.netplay_service.queue_client import RUNNER_PROTOCOL_VERSION
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import QueueProtocolError
from hal.netplay_service.queue_client import SessionState
from hal.netplay_service.queue_client import parse_job
from hal.netplay_service.queue_client import parse_session_state
from hal.netplay_service.queue_contract import SessionEndedError

_MAX_BYTES = 16 * 1024
_HEARTBEAT_SECONDS = 10.0


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], value)


@dataclass(frozen=True, slots=True)
class Observation:
    job_id: str
    attempt: int
    seq: int
    received_at: float
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Snapshot:
    job: Job
    payload: dict[str, object]


class QueueControl:
    """Relay local reports and claims; only durable changes become HTTP requests.

    The loopback endpoint uses the same wire contract as RemoteQueue. Slot
    queue clients use a loopback endpoint and local token. Cloudflare control
    traffic stays in the supervisor.
    """

    def __init__(
        self,
        endpoint: QueueEndpoint,
        session_id: str,
        status: Callable[[], RunnerStatus],
        status_path: Path,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.remote = endpoint
        self.session_id = session_id
        self._status = status
        self._status_path = status_path
        self._http = client or httpx.Client(base_url=endpoint.url, headers=endpoint.headers(), timeout=10)
        self._lock = threading.RLock()
        self._commands = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._socket: ClientConnection | None = None
        self._failure: str | None = None
        self._connected = False
        self._last_health = time.monotonic()
        self._assignments: dict[int, str] = {}
        self._jobs: dict[str, Snapshot] = {}
        self._observations: dict[int, Observation] = {}
        self._capacity: dict[str, object] = {"queued": 0}
        self._state: dict[str, object] | None = None
        self._pairing: object = None
        self._work_signature = ""
        self._work_epoch = 0
        self._claims: dict[int, int] = {}
        self._server = _Server(("127.0.0.1", 0), self)
        self.endpoint = QueueEndpoint(f"http://127.0.0.1:{self._server.server_port}", secrets.token_urlsafe(32))
        self._server_thread = threading.Thread(target=self._server.serve_forever, name="queue-local", daemon=True)
        self._socket_thread = threading.Thread(target=self._run, name="queue-control", daemon=True)

    def __enter__(self) -> QueueControl:
        self._server_thread.start()
        self._socket_thread.start()
        if not self._ready.wait(20) or self._state is None:
            self.__exit__()
            raise QueueProtocolError(self._failure or "queue control connection did not become ready")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._socket is not None:
            self._socket.close()
        if self._server_thread.is_alive():
            self._server.shutdown()
        self._server.server_close()
        if self._socket_thread.is_alive():
            self._socket_thread.join(timeout=25)
        if self._server_thread.is_alive():
            self._server_thread.join(timeout=5)
        self._http.close()

    def state(self) -> SessionState:
        with self._lock:
            if self._failure is not None:
                raise QueueProtocolError(self._failure)
            if time.monotonic() - self._last_health >= 30:
                raise SessionEndedError("session control heartbeat expired")
            if self._state is None:
                raise QueueProtocolError("queue control is not ready")
            return parse_session_state(self._state)

    def _progress(self) -> list[dict[str, object]]:
        with self._lock:
            samples = tuple(self._observations.items())
        result: list[dict[str, object]] = []
        for slot, sample in samples:
            if time.monotonic() - sample.received_at > 6:
                continue
            path = self._status_path.with_name(f"{self._status_path.name}.slot-{slot}.json")
            try:
                health = read_slot_status(path)
            except ValueError:
                continue
            if health.slot != slot or abs(time.time() - health.updated_at) > 5 or health.state is SlotState.RECOVERING:
                continue
            result.append({"slot": slot, "job_id": sample.job_id, "attempt": sample.attempt, "seq": sample.seq})
        return result

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                url = self.remote.url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
                headers = {**self.remote.headers(), "X-HAL-Session": self.session_id}
                with connect(
                    f"{url}/v1/runner/live",
                    additional_headers=headers,
                    max_size=_MAX_BYTES,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=None,
                ) as ws:
                    self._socket = ws
                    self._listen(ws)
                    failures = 0
            except (OSError, WebSocketException, TimeoutError, ValueError, QueueProtocolError) as error:
                # Do not log handshake URLs or headers, which contain service credentials.
                logger.warning("queue control disconnected: {}", type(error).__name__)
                with self._lock:
                    self._connected = False
                if isinstance(error, QueueProtocolError):
                    self._failure = str(error)
                    self._ready.set()
                    return
            if self._stop.wait(min(30, 2 ** min(failures, 5)) * random.uniform(0.75, 1.25)):
                break
            failures += 1

    def _listen(self, ws: ClientConnection) -> None:
        hello = _object(json.loads(ws.recv(timeout=10)))
        if hello.get("protocol_version") != RUNNER_PROTOCOL_VERSION:
            raise QueueProtocolError("queue control protocol mismatch")
        ack = hello["sequence"]
        ws.send(json.dumps({"type": "subscribe", "ack": ack}))
        next_health = 0.0
        last_health = time.monotonic()
        with self._lock:
            self._work_signature = ""
            self._claims.clear()
        pages: list[str] = []
        while not self._stop.is_set():
            if time.monotonic() >= next_health:
                ws.send(
                    json.dumps(
                        {
                            "type": "health",
                            "ack": ack,
                            "status": self._status().to_payload(),
                            "progress": self._progress(),
                        }
                    )
                )
                next_health = time.monotonic() + _HEARTBEAT_SECONDS
            if time.monotonic() - last_health > 20:
                raise TimeoutError("queue health acknowledgement timed out")
            try:
                value = _object(json.loads(ws.recv(timeout=1)))
            except TimeoutError:
                continue
            if value.get("type") == "page":
                text, count = value.get("text"), value.get("count")
                if value.get("page") != len(pages) or not isinstance(count, int) or not 1 <= count <= 128:
                    raise QueueProtocolError("invalid snapshot page")
                if not isinstance(text, str) or len(text) > 2048:
                    raise QueueProtocolError("invalid snapshot page text")
                pages.append(text)
                if len(pages) < count:
                    continue
                value = _object(json.loads("".join(pages)))
                pages = []
            ack = value["sequence"]
            kind = value.get("type")
            with self._lock:
                if kind in ("health", "state"):
                    next_state = {"draining": value["draining"], "stream": value["stream"]}
                    parse_session_state(next_state)
                    self._state = next_state
                    if kind == "health":
                        last_health = time.monotonic()
                        self._last_health = last_health
                        self._connected = True
                        self._ready.set()
                elif kind == "capacity":
                    capacity = _object(value["capacity"])
                    if capacity.get("healthy_slots") != self._capacity.get("healthy_slots"):
                        self._work_epoch += 1
                    self._capacity = capacity
                elif kind == "work":
                    assignments = value["assignments"]
                    if not isinstance(assignments, list):
                        raise QueueProtocolError("invalid assignments snapshot")
                    self._assignments = {}
                    for item in map(_object, assignments):
                        slot, job_id = item.get("slot"), item.get("id")
                        if not isinstance(slot, int) or not isinstance(job_id, str):
                            raise QueueProtocolError("invalid assignment identity")
                        self._assignments[slot] = job_id
                    keep = set(self._assignments.values())
                    self._jobs = {key: snapshot for key, snapshot in self._jobs.items() if key in keep}
                    self._observations = {
                        slot: sample
                        for slot, sample in self._observations.items()
                        if self._assignments.get(slot) == sample.job_id
                    }
                    self._pairing = value["pairing"]
                    signature = json.dumps([self._capacity.get("queued"), self._capacity.get("active"), self._pairing])
                    if signature != self._work_signature:
                        self._work_signature = signature
                        self._work_epoch += 1
                elif kind == "job":
                    self._remember(_object(value["job"]))
                elif kind == "error":
                    raise QueueProtocolError(str(value.get("detail", "queue rejected control message")))
            if isinstance(ack, int) and ack % 32 == 0:
                ws.send(json.dumps({"type": "ack", "ack": ack}))

    def _remember(self, payload: dict[str, object]) -> None:
        job = parse_job(payload)
        previous = self._jobs.get(job.id)
        if previous is not None:
            old = previous.job
            stale = old.attempt > job.attempt or (
                old.attempt == job.attempt
                and (
                    old.settings.revision > job.settings.revision
                    or old.lock_requests > job.lock_requests
                    or (old.wind_down is not None and job.wind_down is None)
                    or max((g.number for g in old.games), default=0) > max((g.number for g in job.games), default=0)
                )
            )
            if stale:
                return
        if job.status is JobStatus.ENDED:
            self._jobs.pop(job.id, None)
            return
        self._jobs[job.id] = Snapshot(job, payload)

    def dispatch(
        self, method: str, path: str, body: object, slot: int | None, attempt: int | None = None
    ) -> tuple[int, object]:
        prefix = f"/v1/runner/sessions/{self.session_id}"
        with self._lock:
            if path == "/v1/capacity" and method == "GET":
                return 200, self._capacity
            if path == f"{prefix}/status" and method == "POST":
                if self._failure is not None:
                    return 409, {"detail": self._failure}
                if not self._connected and time.monotonic() - self._last_health >= 30:
                    return 410, {"detail": "session control heartbeat expired"}
                return (200, self._state) if self._connected else (503, {"detail": "control connection unavailable"})
            if path == f"{prefix}/pairing" and method == "GET":
                return 200, {"pairing": self._pairing}
        if method == "POST" and path == f"{prefix}/claim":
            return self._claim(path, _object(body))
        job_match = re.fullmatch(r"/v1/runner/jobs/([A-Za-z0-9_-]+)(?:/(report|end|replay))?", path)
        if job_match is not None and slot is not None:
            job_id, action = job_match.groups()
            if action in ("report", "end"):
                with self._lock:
                    snapshot = self._jobs.get(job_id)
                    if attempt is None or (snapshot is not None and snapshot.job.attempt != attempt):
                        return 409, {"detail": "job attempt has changed"}
            if method == "POST" and action == "report":
                return self._report(path, job_id, slot, _object(body), attempt)
            if method == "GET" and action is None:
                with self._lock:
                    snapshot = self._jobs.get(job_id)
                    if snapshot is not None:
                        return 200, snapshot.payload
            if (method == "POST" and action in ("end", "replay")) or method == "GET":
                return self._forward(method, path, body, slot, attempt)
        if method == "POST" and path in (f"{prefix}/drain", f"{prefix}/pairing-finished"):
            return self._forward(method, path, body, None)
        return 404, {"detail": "unknown local control route"}

    def _claim(self, path: str, body: dict[str, object]) -> tuple[int, object]:
        slot = body.get("slot")
        if not isinstance(slot, int):
            return 422, {"detail": "slot must be an integer"}
        with self._commands:
            with self._lock:
                if not self._connected:
                    return 503, {"detail": "control connection unavailable"}
                held_id = self._assignments.get(slot)
                held = None if held_id is None else self._jobs.get(held_id)
                if held is not None and held.job.status is JobStatus.ASSIGNED and held.job.phase is None:
                    return 200, held.payload
                if (
                    self._pairing is not None
                    or self._capacity.get("queued") == 0
                    or self._capacity.get("healthy_slots") == 0
                    or self._claims.get(slot) == self._work_epoch
                ):
                    return 204, None
                self._claims[slot] = self._work_epoch
            status, payload = self._forward("POST", path, body, None)
            if status == 200:
                with self._lock:
                    claimed = parse_job(payload)
                    self._assignments[slot] = claimed.id
                    self._pairing = {"slot": slot, "job_id": claimed.id, "attempt": claimed.attempt}
            if status >= 500:
                with self._lock:
                    self._claims.pop(slot, None)
            return status, payload

    def _report(
        self, path: str, job_id: str, slot: int, body: dict[str, object], attempt: int | None
    ) -> tuple[int, object]:
        fingerprint = json.dumps(
            {k: v for k, v in body.items() if k not in ("seq", "phase_seconds_left", "finished_games")}, sort_keys=True
        )
        seq = body.get("seq")
        if not isinstance(seq, int):
            return 422, {"detail": "seq must be an integer"}
        with self._commands:
            with self._lock:
                if time.monotonic() - self._last_health >= 30:
                    return 409, {"detail": "session control heartbeat expired"}
                previous = self._observations.get(slot)
                snapshot = self._jobs.get(job_id)
                if snapshot is None or snapshot.job.attempt != attempt:
                    return 409, {"detail": "reservation attempt is absent from the control snapshot"}
                sample = Observation(job_id, snapshot.job.attempt, seq, time.monotonic(), fingerprint)
                unchanged = (
                    previous is not None
                    and previous.job_id == job_id
                    and previous.attempt == sample.attempt
                    and previous.fingerprint == fingerprint
                    and not body.get("finished_games")
                )
                if unchanged:
                    self._observations[slot] = sample
                    return 200, snapshot.payload
            known_games = {game.number for game in snapshot.job.games}
            games = body.get("finished_games")
            if not isinstance(games, list):
                return 422, {"detail": "finished_games must be a list"}
            delta = [game for game in map(_object, games) if game.get("number") not in known_games]
            status, payload = self._forward(
                "POST", path, {**body, "finished_games": delta}, slot, snapshot.job.attempt
            )
            if status == 200:
                with self._lock:
                    self._observations[slot] = sample
            return status, payload

    def _forward(
        self, method: str, path: str, body: object, slot: int | None, attempt: int | None = None
    ) -> tuple[int, object]:
        headers = {"X-HAL-Session": self.session_id}
        if attempt is not None:
            headers["X-HAL-Attempt"] = str(attempt)
        if slot is not None:
            headers["X-HAL-Slot"] = str(slot)
        try:
            response = self._http.request(method, path, json=body, headers=headers)
        except httpx.TransportError:
            return 503, {"detail": "queue command unavailable"}
        try:
            payload: object = None if response.status_code == 204 else response.json()
        except ValueError:
            return 503, {"detail": "queue returned an invalid response"}
        if (
            response.status_code in (200, 201)
            and isinstance(payload, dict)
            and "id" in payload
            and not path.endswith("/replay")
        ):
            with self._lock:
                body = _object(payload)
                self._remember(body)
                current = self._jobs.get(str(body["id"]))
                if current is not None:
                    payload = current.payload
        return response.status_code, payload


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], control: QueueControl) -> None:
        self.control = control
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        control = self.server.control
        if not secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {control.endpoint.token}"):
            self.send_error(401)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= _MAX_BYTES:
                self.send_error(413)
                return
            body: object = json.loads(self.rfile.read(length)) if length else None
            raw_slot = self.headers.get("X-HAL-Slot")
            slot = None if raw_slot is None else int(raw_slot)
            raw_attempt = self.headers.get("X-HAL-Attempt")
            attempt = None if raw_attempt is None else int(raw_attempt)
            status, result = control.dispatch(self.command, self.path, body, slot, attempt)
        except ValueError, QueueProtocolError:
            status, result = 422, {"detail": "invalid local control request"}
        data = b"" if result is None else json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
