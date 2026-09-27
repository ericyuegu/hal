"""Run a local, two-account 059 netplay qualification through the service runner.

The runner owns a separate GPU process and Dolphin slot. A second local Dolphin
drives a neutral peer. Qualification replays stay on disk; replay publication is
deferred so this command cannot write to the production R2 bucket.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import cast

import melee

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import POLICY_BUTTON_MASK
from hal.eval.scheduling import FrameTiming
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobCredentials
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.domain import validate_player_code
from hal.netplay_service.health import ChunkHealth
from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import read_runner_status
from hal.netplay_service.queue import QueueStore
from hal.netplay_service.runner import RunnerConfig
from hal.netplay_service.runner import run as run_netplay_service
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.sim.netplay import NetplaySession
from hal.sim.netplay import NetplaySetup


@dataclass(frozen=True, slots=True)
class QualificationConfig:
    output: Path
    bundle: Path
    bot_account: Path
    peer_account: Path
    delay: int
    desired_return: float
    minimum_games: int
    minimum_gameplay_seconds: float
    smoke_frames: int | None
    bot_slippi_port: int
    peer_slippi_port: int


@dataclass(frozen=True, slots=True)
class PeerGame:
    reservation_id: str
    game_number: int
    frames: int
    first_frame: int
    last_frame: int
    ended: bool
    step_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ResourceSample:
    elapsed_seconds: float
    process_count: int
    thread_count: int
    rss_kib: int
    gpu_memory_mib: int | None
    runner_state: str | None


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    seconds: float
    forced_runner: bool
    forced_descendants: tuple[int, ...]
    remaining_descendants: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MatchAssessment:
    measurement: str
    measurement_sha256: str
    gameplay_seconds: float
    steady_start_frame: int
    steady_frame_count: int
    steady_game_fps: float
    steady_inference_count: int
    plan_decision_count: int
    accepted_plan_count: int
    steady_inference_p95_ms: float
    steady_inference_p99_ms: float
    startup_count_by_event: dict[str, int]
    steady_count_by_event: dict[str, int]
    failures: tuple[str, ...]


_STEADY_START_FRAME = 300
_SCHEDULE_COUNTERS = (
    "deadline_misses",
    "prefix_mismatches",
    "exhausted_chunks",
    "neutral_fallback_frames",
    "submission_gaps",
    "transport_corrections",
)
_UNMEASURED_GATES = (
    "Post-preparation compilation and CUDA graph capture counts",
    "Process, thread, and memory stability across rematches",
    "Three same-day matched control/candidate delivery trials",
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[object], value)


def _wire_prefix(value: object, name: str, *, allow_missing: bool) -> tuple[tuple[int, ...] | None, ...]:
    prefix = []
    for action in _array(value, name):
        if action is None and allow_missing:
            prefix.append(None)
            continue
        channels = tuple(_integer(channel, name) for channel in _array(action, name))
        if len(channels) != 7:
            raise ValueError(f"{name} must contain seven controller wire values per action")
        if (
            any(not -80 <= channel <= 80 for channel in channels[:4])
            or any(not 0 <= channel <= 140 for channel in channels[4:6])
            or channels[6] < 0
            or channels[6] & ~POLICY_BUTTON_MASK
        ):
            raise ValueError(f"{name} contains invalid controller wire values")
        prefix.append(channels)
    return tuple(prefix)


def _validate_plan_decisions(
    payload: Mapping[str, object],
    inference_frames: tuple[int, ...],
    timing: FrameTiming,
    last_choice_frame: int,
    path: Path,
) -> int:
    decisions = _array(payload.get("plan_decisions"), "plan_decisions")
    if len(decisions) != len(inference_frames):
        raise ValueError(f"{path}: plan decisions do not cover every inference response")
    stream_id = _integer(payload.get("stream_id"), "stream_id")
    generation = _integer(payload.get("generation"), "generation")
    if generation < 1:
        raise ValueError(f"{path}: match generation must be positive")
    accepted_count = 0
    previous_choice: int | None = None
    for index, (raw, source_frame) in enumerate(zip(decisions, inference_frames, strict=True)):
        decision = _mapping(raw, "plan decision")
        request_identity = tuple(
            _integer(decision.get(f"request_{name}"), f"request_{name}")
            for name in ("stream_id", "generation", "sequence", "source_frame")
        )
        response_identity = tuple(
            _integer(decision.get(f"response_{name}"), f"response_{name}")
            for name in ("stream_id", "generation", "sequence", "source_frame")
        )
        if (
            request_identity != response_identity
            or request_identity[0] != stream_id
            or request_identity[1] != generation
            or request_identity[2] != index
            or request_identity[3] != source_frame
        ):
            raise ValueError(f"{path}: plan decision identity differs from its inference response")
        choice_frame = _integer(decision.get("choice_frame"), "plan choice frame")
        if (
            choice_frame < source_frame
            or choice_frame > last_choice_frame
            or (previous_choice is not None and choice_frame <= previous_choice)
        ):
            raise ValueError(f"{path}: plan decisions have invalid controller choice frames")
        previous_choice = choice_frame
        first_target = choice_frame + timing.physical_delay_frames + 1
        if _integer(decision.get("first_submittable_target"), "first submittable target") != first_target:
            raise ValueError(f"{path}: plan decision submitability bound differs from frame timing")
        targets = tuple(
            _integer(value, "generated target frame")
            for value in _array(decision.get("generated_target_frames"), "generated_target_frames")
        )
        expected_targets = tuple(
            range(source_frame + timing.fixed_prefix_frames + 1, source_frame + timing.prediction_horizon_frames + 1)
        )
        if targets != expected_targets:
            raise ValueError(f"{path}: plan decision target frames differ from the declared horizon")
        requested = _wire_prefix(decision.get("request_prefix_wire"), "request_prefix_wire", allow_missing=False)
        pinned = _wire_prefix(decision.get("pinned_prefix_wire"), "pinned_prefix_wire", allow_missing=True)
        if len(requested) != timing.fixed_prefix_frames or len(pinned) != timing.fixed_prefix_frames:
            raise ValueError(f"{path}: plan decision fixed prefix differs from the declared shape")
        accepted = decision.get("accepted")
        if type(accepted) is not bool:
            raise ValueError(f"{path}: plan decision acceptance must be a boolean")
        valid = requested == pinned and all(target >= first_target for target in targets)
        if accepted != valid:
            raise ValueError(f"{path}: accepted plan is late or has a mismatched wire prefix")
        accepted_count += int(accepted)
    return accepted_count


def _assess_match(
    path: Path,
    *,
    reservation_id: str,
    game_number: int,
    delay: int,
    desired_return: float,
    bundle_sha256: str,
    source_git_sha: str,
) -> MatchAssessment:
    payload = _mapping(json.loads(path.read_text()), "match measurement")
    timing = FrameTiming(delay, 1, delay + 1, 4, 8)
    expected = {
        "schema_version": 2,
        "reservation_id": reservation_id,
        "game_number": game_number,
        "policy_bundle_sha256": bundle_sha256,
        "source_git_sha": source_git_sha,
        "desired_return": desired_return,
        "observation_mode": "first_seen_speculative",
        "timing": asdict(timing),
    }
    if any(payload.get(key) != value for key, value in expected.items()) or "failure" in payload:
        raise ValueError(f"{path}: match identity or timing profile differs from the qualification run")
    health = ChunkHealth.from_payload(payload.get("schedule"))
    if health.schedule != timing:
        raise ValueError(f"{path}: scheduling counters use a different timing profile")
    frames = tuple(_integer(value, "frame ID") for value in _array(payload.get("frame_ids"), "frame_ids"))
    intervals = tuple(
        _number(value, "frame interval")
        for value in _array(payload.get("frame_interval_seconds"), "frame_interval_seconds")
    )
    # The last capture is the menu frame that ends play, not another gameplay frame.
    if len(frames) != len(intervals) + 1 or len(frames) < 3 or frames[:-1] != tuple(range(len(frames) - 1)):
        raise ValueError(f"{path}: frame intervals do not cover consecutive gameplay observations")
    inference_frames = tuple(
        _integer(value, "inference source frame")
        for value in _array(payload.get("inference_source_frames"), "inference_source_frames")
    )
    inference_seconds = tuple(
        _number(value, "inference seconds") for value in _array(payload.get("inference_seconds"), "inference_seconds")
    )
    if len(inference_frames) != len(inference_seconds) or any(
        following <= previous for previous, following in zip(inference_frames, inference_frames[1:], strict=False)
    ):
        raise ValueError(f"{path}: inference timings have missing or repeated source frames")
    if any(frame > frames[-2] for frame in inference_frames):
        raise ValueError(f"{path}: inference source frame follows the last controller choice")
    accepted_plan_count = _validate_plan_decisions(payload, inference_frames, timing, frames[-2], path)

    startup_counts = dict.fromkeys(_SCHEDULE_COUNTERS, 0)
    previous_counts = startup_counts.copy()
    previous_frame: int | None = None
    events = _array(payload.get("schedule_events"), "schedule_events")
    if not events:
        raise ValueError(f"{path}: missing scheduling events")
    inference_failed = False
    for raw in events:
        event = _mapping(raw, "schedule event")
        frame = _integer(event.get("choice_frame"), "choice frame")
        if previous_frame is not None and frame <= previous_frame:
            raise ValueError(f"{path}: scheduling event frames are not increasing")
        if event.get("phase") != ("countdown" if frame < 0 else "gameplay"):
            raise ValueError(f"{path}: scheduling event has an incorrect phase")
        if event.get("target_frame") != frame + delay + 1 or frame > frames[-2]:
            raise ValueError(f"{path}: scheduling event has an incorrect controller target")
        counts = {name: _integer(event.get(name), name) for name in _SCHEDULE_COUNTERS}
        if any(counts[name] < previous_counts[name] for name in _SCHEDULE_COUNTERS):
            raise ValueError(f"{path}: scheduling counters decreased")
        if not isinstance(event.get("inference_failed"), bool):
            raise ValueError(f"{path}: missing inference failure status")
        inference_failed = inference_failed or event["inference_failed"] is True
        if frame < _STEADY_START_FRAME:
            startup_counts = counts
        previous_counts = counts
        previous_frame = frame
    if previous_counts != {name: getattr(health, name) for name in _SCHEDULE_COUNTERS}:
        raise ValueError(f"{path}: final scheduling counters differ from the event trace")
    if payload.get("controller_submission_gaps") != health.submission_gaps:
        raise ValueError(f"{path}: submission gap totals differ")
    steady_counts = {name: previous_counts[name] - startup_counts[name] for name in _SCHEDULE_COUNTERS}
    steady_intervals = intervals[_STEADY_START_FRAME:-1]
    steady_inference = sorted(
        seconds
        for frame, seconds in zip(inference_frames, inference_seconds, strict=True)
        if frame >= _STEADY_START_FRAME
    )
    if not steady_intervals or sum(steady_intervals) <= 0 or not steady_inference:
        raise ValueError(f"{path}: no steady gameplay or inference measurements after frame {_STEADY_START_FRAME}")
    fps = len(steady_intervals) / sum(steady_intervals)
    p95 = steady_inference[math.ceil(0.95 * len(steady_inference)) - 1]
    p99 = steady_inference[math.ceil(0.99 * len(steady_inference)) - 1]
    failures = []
    if fps < 59.5:
        failures.append("steady gameplay below 59.5 FPS")
    if delay == 2 and p95 > 0.012:
        failures.append("steady inference delivery p95 exceeds 12 ms")
    if p99 >= 1 / 60:
        failures.append("steady inference delivery p99 reaches the one-frame allowance")
    if health.submission_gaps:
        failures.append("controller submission frames were skipped")
    if steady_counts["exhausted_chunks"] or steady_counts["neutral_fallback_frames"]:
        failures.append("action plans were exhausted or neutral fallback occurred after startup")
    if inference_failed:
        failures.append("inference failed during the match")
    return MatchAssessment(
        str(path),
        _sha256(path),
        _number(payload.get("gameplay_seconds"), "gameplay seconds"),
        _STEADY_START_FRAME,
        len(steady_intervals),
        fps,
        len(steady_inference),
        len(inference_frames),
        accepted_plan_count,
        p95 * 1000,
        p99 * 1000,
        startup_counts,
        steady_counts,
        tuple(failures),
    )


def _write_json(path: Path, payload: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, allow_nan=False, sort_keys=True, indent=2))
    pending.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((*(root / "hal").rglob("*.py"), Path(__file__).resolve()))
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _account_code(path: Path) -> str:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("connectCode"), str):
        raise ValueError(f"invalid Slippi account file {path}")
    return validate_player_code(payload["connectCode"])


def _proc_tree(root_pid: int) -> tuple[int, ...]:
    found: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found or not Path(f"/proc/{pid}").exists():
            continue
        found.add(pid)
        children_path = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(value) for value in children_path.read_text().split())
        except OSError:
            continue
    return tuple(sorted(found))


def _process_identity(pid: int) -> tuple[str, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], int(fields[19])
    except IndexError, OSError, ValueError:
        return None


def _live_owned_processes(owned: Mapping[int, int]) -> tuple[int, ...]:
    return tuple(
        pid
        for pid, started in owned.items()
        if (identity := _process_identity(pid)) is not None and identity[0] != "Z" and identity[1] == started
    )


def _status_counts(pid: int) -> tuple[int, int]:
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:
        return 0, 0
    values = dict(line.split(":", 1) for line in status if ":" in line)
    return int(values.get("Threads", "0").strip()), int(values.get("VmRSS", "0 kB").split()[0])


def _gpu_memory(pids: tuple[int, ...]) -> int | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout
    except OSError, ValueError, subprocess.SubprocessError:
        return None
    if not output.strip():
        return 0
    total = 0
    matched = False
    for line in output.splitlines():
        parts = line.split(",", 1)
        if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
            return None
        pid, memory_mib = (int(part.strip()) for part in parts)
        if pid in pids:
            total += memory_mib
            matched = True
    # nvidia-smi can report host PIDs while /proc exposes container PIDs.
    return total if matched else None


class _ResourceSampler:
    def __init__(self, root_pid: int, status_path: Path) -> None:
        self.root_pid = root_pid
        self.status_path = status_path
        self.started = time.monotonic()
        self.samples: list[ResourceSample] = []
        self.owned_processes: dict[int, int] = {}
        self._index = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._collect, name="hal-netplay-qualification-sampler")

    def __enter__(self) -> _ResourceSampler:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        # A GPU sample can be inside the three-second nvidia-smi timeout.
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            raise RuntimeError("qualification resource sampler did not stop")

    def _collect(self) -> None:
        while not self.stop.is_set():
            self.sample()
            self.stop.wait(1)

    def sample(self) -> None:
        pids = _proc_tree(self.root_pid)
        for pid in pids:
            identity = _process_identity(pid)
            if identity is not None:
                self.owned_processes[pid] = identity[1]
        counts = tuple(_status_counts(pid) for pid in pids)
        state: str | None = None
        with suppress(OSError, ValueError):
            state = read_runner_status(self.status_path).state.value
        self.samples.append(
            ResourceSample(
                time.monotonic() - self.started,
                len(pids),
                sum(item[0] for item in counts),
                sum(item[1] for item in counts),
                _gpu_memory(pids) if self._index % 30 == 0 else None,
                state,
            )
        )
        self._index += 1


def _wait_ready(process: BaseProcess, status_path: Path, timeout_seconds: float) -> float:
    started = time.monotonic()
    deadline = started + timeout_seconds
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"netplay service exited before readiness: {process.exitcode}")
        try:
            status = read_runner_status(status_path)
        except OSError, ValueError:
            time.sleep(0.2)
            continue
        if status.state is RunnerState.READY:
            return time.monotonic() - started
        if status.state is RunnerState.UNAVAILABLE:
            raise RuntimeError(f"netplay service unavailable: {status.message}")
        time.sleep(0.2)
    raise TimeoutError(f"netplay service did not prepare within {timeout_seconds:g} seconds")


def _wait_job(store: QueueStore, credentials: JobCredentials, process: BaseProcess, timeout_seconds: float) -> Job:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process.is_alive():
            raise RuntimeError(f"netplay service exited during a reservation: {process.exitcode}")
        job = store.get_job(credentials.job.id, credentials.token)
        if job.status in (JobStatus.REMATCH_WAIT, JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELED):
            return job
        time.sleep(0.1)
    raise TimeoutError("reservation did not finish or enter rematch wait")


def _play_peer_game(
    session: NetplaySession,
    first: dict,
    credentials: JobCredentials,
    game_number: int,
    smoke_frames: int | None,
    *,
    store: QueueStore,
    runner: BaseProcess,
) -> PeerGame:
    first_id = int(first["id"])
    last_id = first_id
    steps: list[float] = []
    next_health_check = time.monotonic() + 1
    while True:
        started = time.perf_counter()
        frame, in_game = session.step(NEUTRAL_CONTROLLER_ACTION)
        steps.append(time.perf_counter() - started)
        if not in_game:
            return PeerGame(credentials.job.id, game_number, len(steps), first_id, last_id, True, tuple(steps))
        last_id = int(frame["id"])
        is_smoke_complete = smoke_frames is not None and last_id >= smoke_frames
        if time.monotonic() >= next_health_check or is_smoke_complete:
            _validate_live_reservation(
                store.get_job(credentials.job.id, credentials.token), is_runner_alive=runner.is_alive()
            )
            next_health_check = time.monotonic() + 1
        if is_smoke_complete:
            return PeerGame(credentials.job.id, game_number, len(steps), first_id, last_id, False, tuple(steps))


def _validate_live_reservation(job: Job, *, is_runner_alive: bool) -> None:
    if not is_runner_alive:
        raise RuntimeError("netplay service exited while the peer was still playing")
    if job.status not in (JobStatus.PLAYING, JobStatus.REMATCH_WAIT, JobStatus.COMPLETE):
        raise RuntimeError(
            f"bot reservation stopped while the peer was still playing: {job.status.value}: {job.error_code}"
        )


def _terminate(process: BaseProcess, observed_processes: Mapping[int, int]) -> ShutdownResult:
    started = time.monotonic()
    pid = process.pid
    owned = {owned_pid: start for owned_pid, start in observed_processes.items() if owned_pid != os.getpid()}
    if pid is not None:
        for child_pid in _proc_tree(pid):
            identity = _process_identity(child_pid)
            if identity is not None:
                owned[child_pid] = identity[1]
    forced_runner = False
    if process.is_alive() and pid is not None:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
        # The runner has its own two-second child termination deadline.
        process.join(timeout=4)
    if process.is_alive():
        forced_runner = True
        process.kill()
        process.join(timeout=1)
    elif pid is not None:
        process.join(timeout=0)

    forced_descendants = _live_owned_processes(owned)
    for child_pid in forced_descendants:
        with suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
    deadline = time.monotonic() + 1
    remaining = _live_owned_processes(owned)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.02)
        remaining = _live_owned_processes(owned)
    if process.is_alive() and pid is not None:
        remaining = (*remaining, pid)
    return ShutdownResult(time.monotonic() - started, forced_runner, forced_descendants, remaining)


def qualify(config: QualificationConfig) -> dict[str, object]:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("netplay qualification requires a display; run it with xvfb-run -a")
    if config.bot_slippi_port == config.peer_slippi_port or any(
        port < 1 or port > 65535 for port in (config.bot_slippi_port, config.peer_slippi_port)
    ):
        raise ValueError("qualification requires distinct valid Slippi ports")
    config.output.mkdir(parents=True, exist_ok=False)
    bot_code = _account_code(config.bot_account)
    peer_code = _account_code(config.peer_account)
    if bot_code == peer_code or config.bot_account.samefile(config.peer_account):
        raise ValueError("qualification requires two distinct Slippi accounts")
    bundle_sha256 = _sha256(config.bundle)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    source_hashes = _source_hashes()
    manifest = {
        "schema_version": 1,
        "workload": f"netplay-059-delay-{config.delay}-neutral-peer",
        "topology": "two local Slippi accounts; separate runner GPU and Dolphin workers; local neutral-input peer",
        "interpretation": "functional and real-time transport qualification only; no gameplay-strength claim",
        "source_git_sha": source_sha,
        "source_file_sha256": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: _version(name) for name in ("torch", "melee", "peppi-py", "mosaicml-streaming", "slpz")
        },
        "pyproject_sha256": _sha256(Path(__file__).resolve().parents[1] / "pyproject.toml"),
        "uv_lock_sha256": _sha256(Path(__file__).resolve().parents[1] / "uv.lock"),
        "bundle": str(config.bundle.resolve()),
        "bundle_sha256": bundle_sha256,
        "bot_account_sha256": _sha256(config.bot_account),
        "peer_account_sha256": _sha256(config.peer_account),
        "iso_sha256": _sha256(Path(ISO_PATH)),
        "dolphin_sha256": _sha256(Path(NETPLAY_EMULATOR_PATH)),
        "delay": config.delay,
        "desired_return": config.desired_return,
        "minimum_games": config.minimum_games,
        "minimum_gameplay_seconds": config.minimum_gameplay_seconds,
        "smoke_frames": config.smoke_frames,
        "bot_slippi_port": config.bot_slippi_port,
        "peer_slippi_port": config.peer_slippi_port,
        "display": os.environ["DISPLAY"],
        "replay_publication": "deferred; local replays retained",
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(config.output / "manifest.json", manifest)
    store = QueueStore(config.output / "queue.sqlite3")
    status_path = config.output / "runner-status.json"
    runner = RunnerConfig(
        database=store.path,
        policy=config.bundle,
        user_jsons=(config.bot_account,),
        slippi_ports=(config.bot_slippi_port,),
        iso_path=Path(ISO_PATH),
        dolphin_path=Path(NETPLAY_EMULATOR_PATH),
        replay_dir=config.output / "runner-replays",
        status_path=status_path,
        git_sha=source_sha,
        device="cuda",
        seed=0,
        compiled=True,
        batch_wait_seconds=0.0005,
        measurement_dir=config.output / "match-measurements",
        publish_replays=False,
    )
    process = mp.get_context("spawn").Process(target=run_netplay_service, args=(runner,), name="hal-059-qualification")
    games: list[PeerGame] = []
    assessments: list[MatchAssessment] = []
    startup_seconds: float | None = None
    failure: str | None = None
    primary_error: BaseException | None = None
    gameplay_seconds = 0.0
    sampler: _ResourceSampler | None = None
    try:
        process.start()
        if process.pid is None:
            raise RuntimeError("qualification service did not start")
        sampler = _ResourceSampler(os.getpid(), status_path)
        with sampler:
            startup_seconds = _wait_ready(process, status_path, runner.preparation_timeout_seconds + 30)
            while True:
                credentials = store.create_job(
                    peer_code,
                    MatchChoices("FOX", "IBDW#0", config.delay, desired_return=config.desired_return),
                )
                replay_dir = config.output / "peer-replays" / credentials.job.id
                replay_dir.mkdir(parents=True)
                with NetplaySession(
                    ISO_PATH,
                    dolphin_path=NETPLAY_EMULATOR_PATH,
                    user_json_path=config.peer_account,
                    online_delay=config.delay,
                    replay_dir=replay_dir,
                    slippi_port=config.peer_slippi_port,
                    realtime=True,
                ) as peer:
                    setup = NetplaySetup(melee.Character.FOX, bot_code, costume=1)
                    first = peer.start_match(setup)
                    game_number = 1
                    while True:
                        game = _play_peer_game(
                            peer, first, credentials, game_number, config.smoke_frames, store=store, runner=process
                        )
                        games.append(game)
                        _write_json(config.output / f"peer-game-{len(games):03d}.json", asdict(game))
                        if not game.ended:
                            break
                        job = _wait_job(store, credentials, process, 60)
                        if job.status not in (JobStatus.REMATCH_WAIT, JobStatus.COMPLETE):
                            raise RuntimeError(f"reservation ended as {job.status.value}: {job.error_code}")
                        measurement = config.output / "match-measurements" / f"{job.id}-game-{job.game_count}.json"
                        assessment = _assess_match(
                            measurement,
                            reservation_id=job.id,
                            game_number=game_number,
                            delay=config.delay,
                            desired_return=config.desired_return,
                            bundle_sha256=bundle_sha256,
                            source_git_sha=source_sha,
                        )
                        assessments.append(assessment)
                        gameplay_seconds += assessment.gameplay_seconds
                        if assessment.failures:
                            raise RuntimeError(f"match failed measured qualification checks: {assessment.failures}")
                        if job.status is JobStatus.COMPLETE:
                            break
                        assert job.actual_stage is not None
                        store.request_rematch(
                            job.id,
                            credentials.token,
                            character="FOX",
                            imitation="IBDW#0",
                            stage=job.actual_stage,
                        )
                        setup = NetplaySetup(
                            melee.Character.FOX, bot_code, costume=1, stage=melee.Stage[job.actual_stage]
                        )
                        game_number = job.game_count + 1
                        first = peer.start_rematch(setup)
                is_complete = (
                    len(games) >= config.minimum_games and gameplay_seconds >= config.minimum_gameplay_seconds
                )
                if config.smoke_frames is not None or is_complete:
                    break
    except BaseException as error:
        primary_error = error
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        observed_processes = {} if sampler is None else sampler.owned_processes
        shutdown = _terminate(process, observed_processes)
        finalization_error: RuntimeError | None = None
        sources_unchanged = _source_hashes() == source_hashes
        if shutdown.remaining_descendants:
            finalization_error = RuntimeError(
                f"qualification descendants remain alive: {shutdown.remaining_descendants}"
            )
        elif not sources_unchanged:
            finalization_error = RuntimeError("runtime sources changed during qualification")
        if finalization_error is not None:
            if failure is None:
                failure = str(finalization_error)
            elif primary_error is not None:
                primary_error.add_note(str(finalization_error))
        if sampler is not None:
            sampler.sample()
        samples = [] if sampler is None else [asdict(sample) for sample in sampler.samples]
        _write_json(config.output / "resources.json", samples)
        has_soak_duration = len(assessments) >= 10 and gameplay_seconds >= 1800
        report: dict[str, object] = {
            "schema_version": 2,
            "qualification_status": "failed" if failure is not None else "incomplete",
            "smoke": config.smoke_frames is not None,
            "measured_checks_passed": failure is None and config.smoke_frames is None and has_soak_duration,
            "unmeasured_gates": [
                *_UNMEASURED_GATES,
                *([] if has_soak_duration else ["At least ten completed matches and 1800 gameplay seconds"]),
            ],
            "match_assessments": [asdict(assessment) for assessment in assessments],
            "ended_at": datetime.now(UTC).isoformat(),
            "games_observed": len(games),
            "gameplay_seconds": gameplay_seconds,
            "startup_seconds": startup_seconds,
            "failure": failure,
            "runner_exitcode": process.exitcode,
            "shutdown": asdict(shutdown),
            "source_files_unchanged": sources_unchanged,
        }
        _write_json(config.output / "run-result.json", report)
        if finalization_error is not None and primary_error is None:
            raise finalization_error
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--bot-account", type=Path, required=True)
    parser.add_argument("--peer-account", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=int, choices=(2, 3), default=2)
    parser.add_argument("--desired-return", type=float, default=19.9760597229004)
    parser.add_argument("--minimum-games", type=int, default=10)
    parser.add_argument("--minimum-gameplay-seconds", type=float, default=1800)
    parser.add_argument("--smoke-frames", type=int)
    parser.add_argument("--bot-slippi-port", type=int, default=51451)
    parser.add_argument("--peer-slippi-port", type=int, default=51452)
    args = parser.parse_args(argv)
    if args.minimum_games < 1 or args.minimum_gameplay_seconds < 0:
        parser.error("minimum games and gameplay time must be non-negative")
    if args.smoke_frames is not None and args.smoke_frames < 1:
        parser.error("smoke frame count must be positive")
    result = qualify(
        QualificationConfig(
            args.output,
            args.bundle,
            args.bot_account,
            args.peer_account,
            args.delay,
            args.desired_return,
            args.minimum_games,
            args.minimum_gameplay_seconds,
            args.smoke_frames,
            args.bot_slippi_port,
            args.peer_slippi_port,
        )
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
