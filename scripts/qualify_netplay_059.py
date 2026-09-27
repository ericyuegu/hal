"""Run a local, two-account 059 netplay qualification through the service runner.

The runner owns a separate GPU process and Dolphin slot. A second local Dolphin
drives a neutral peer. Qualification replays stay on disk; replay publication is
deferred so this command cannot write to the production R2 bucket.
"""

import argparse
import hashlib
import importlib.metadata
import json
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

import melee

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobCredentials
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.domain import validate_player_code
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
    total = 0
    for line in output.splitlines():
        parts = line.split(",", 1)
        if len(parts) == 2 and parts[0].strip().isdigit() and int(parts[0]) in pids:
            total += int(parts[1].strip())
    return total


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
        self.thread.join(timeout=2)

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
    raise TimeoutError("netplay service did not prepare within 150 seconds")


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
) -> PeerGame:
    first_id = int(first["id"])
    last_id = first_id
    steps: list[float] = []
    while True:
        started = time.perf_counter()
        frame, in_game = session.step(NEUTRAL_CONTROLLER_ACTION)
        steps.append(time.perf_counter() - started)
        if not in_game:
            return PeerGame(credentials.job.id, game_number, len(steps), first_id, last_id, True, tuple(steps))
        last_id = int(frame["id"])
        if smoke_frames is not None and last_id >= smoke_frames:
            return PeerGame(credentials.job.id, game_number, len(steps), first_id, last_id, False, tuple(steps))


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
            startup_seconds = _wait_ready(process, status_path, 150)
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
                    while True:
                        game_number = len(games) + 1
                        game = _play_peer_game(peer, first, credentials, game_number, config.smoke_frames)
                        games.append(game)
                        _write_json(config.output / f"peer-game-{game_number:03d}.json", asdict(game))
                        if not game.ended:
                            return {"smoke": True, "frames": game.frames, "startup_seconds": startup_seconds}
                        job = _wait_job(store, credentials, process, 60)
                        if job.status not in (JobStatus.REMATCH_WAIT, JobStatus.COMPLETE):
                            raise RuntimeError(f"reservation ended as {job.status.value}: {job.error_code}")
                        measurement = config.output / "match-measurements" / f"{job.id}-game-{job.game_count}.json"
                        payload = json.loads(measurement.read_text())
                        gameplay_seconds += float(payload["gameplay_seconds"])
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
                        first = peer.start_rematch(setup)
                if len(games) >= config.minimum_games and gameplay_seconds >= config.minimum_gameplay_seconds:
                    break
            return {
                "smoke": False,
                "games": len(games),
                "gameplay_seconds": gameplay_seconds,
                "startup_seconds": startup_seconds,
            }
    except BaseException as error:
        primary_error = error
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        observed_processes = {} if sampler is None else sampler.owned_processes
        shutdown = _terminate(process, observed_processes)
        cleanup_error: RuntimeError | None = None
        if shutdown.remaining_descendants:
            cleanup_error = RuntimeError(f"qualification descendants remain alive: {shutdown.remaining_descendants}")
            if failure is None:
                failure = str(cleanup_error)
            elif primary_error is not None:
                primary_error.add_note(str(cleanup_error))
        if sampler is not None:
            sampler.sample()
        samples = [] if sampler is None else [asdict(sample) for sample in sampler.samples]
        _write_json(config.output / "resources.json", samples)
        _write_json(
            config.output / "run-result.json",
            {
                "schema_version": 1,
                "ended_at": datetime.now(UTC).isoformat(),
                "games_observed": len(games),
                "gameplay_seconds": gameplay_seconds,
                "startup_seconds": startup_seconds,
                "failure": failure,
                "runner_exitcode": process.exitcode,
                "shutdown": asdict(shutdown),
                "source_files_unchanged": _source_hashes() == source_hashes,
            },
        )
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error


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
