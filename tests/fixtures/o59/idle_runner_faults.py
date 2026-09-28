"""Measure failure detection and one bounded recovery in an idle netplay runner."""

import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import signal
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack
from contextlib import suppress
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from multiprocessing.process import BaseProcess
from pathlib import Path

from hal.netplay_service.admin import policy_config_for
from hal.netplay_service.assets import account_key
from hal.netplay_service.assets import sha256_file
from hal.netplay_service.domain import account_connect_code
from hal.netplay_service.health import RunnerState
from hal.netplay_service.health import read_runner_status
from hal.netplay_service.local_worker import DEV_ADMIN_TOKEN
from hal.netplay_service.local_worker import DEV_RUNNER_TOKEN
from hal.netplay_service.local_worker import free_port
from hal.netplay_service.local_worker import local_worker
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import RunnerClient
from hal.netplay_service.queue_client import SessionReporter
from hal.netplay_service.queue_client import new_session_id
from hal.netplay_service.runner import RunnerConfig
from hal.netplay_service.runner import SessionStatus
from hal.netplay_service.runner import run
from hal.paths import ISO_PATH
from hal.paths import NETPLAY_EMULATOR_PATH
from hal.training.runs import source_git_sha


@dataclass(frozen=True, slots=True)
class Cleanup:
    seconds: float
    forced_runner: bool
    forced_descendants: tuple[int, ...]
    remaining_descendants: tuple[int, ...]


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _write_json(path: Path, payload: object) -> None:
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    pending.replace(path)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((*(root / "hal").rglob("*.py"), Path(__file__).resolve()))
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def _process_identity(pid: int) -> tuple[str, int] | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[0], int(fields[19])
    except IndexError, OSError, ValueError:
        return None


def _proc_tree(root_pid: int) -> tuple[int, ...]:
    found: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found or not Path(f"/proc/{pid}").exists():
            continue
        found.add(pid)
        try:
            pending.extend(int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
        except OSError:
            continue
    return tuple(sorted(found))


def _remember_descendants(root_pid: int, owned: dict[int, int]) -> None:
    for pid in _proc_tree(root_pid):
        identity = _process_identity(pid)
        if identity is not None:
            owned[pid] = identity[1]


def _live_owned(owned: Mapping[int, int]) -> tuple[int, ...]:
    return tuple(
        pid
        for pid, started in owned.items()
        if (identity := _process_identity(pid)) is not None and identity[0] != "Z" and identity[1] == started
    )


def _has_gpu_device(pid: int, proc_root: Path = Path("/proc")) -> bool:
    # Modal's gVisor reports GPU PIDs in a different namespace from /proc.
    # Open device handles identify the owner inside the runner's process tree.
    try:
        descriptors = tuple((proc_root / str(pid) / "fd").iterdir())
    except FileNotFoundError:
        return False
    for descriptor in descriptors:
        try:
            target = descriptor.readlink()
        except FileNotFoundError:
            continue
        if re.fullmatch(r"/dev/nvidia\d+", str(target)):
            return True
    return False


def _gpu_child(root_pid: int, owned: dict[int, int]) -> int | None:
    _remember_descendants(root_pid, owned)
    candidates = {pid for pid in _proc_tree(root_pid) if _has_gpu_device(pid)}
    if len(candidates) > 1:
        raise RuntimeError(f"runner owns multiple simultaneous GPU processes: {sorted(candidates)}")
    return next(iter(candidates), None)


def _ready(status_path: Path) -> bool:
    if not status_path.exists():
        return False
    try:
        status = read_runner_status(status_path)
    except ValueError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return False
        raise
    if status.state is RunnerState.UNAVAILABLE:
        raise RuntimeError(f"isolated runner became unavailable: {status.message}")
    return status.state is RunnerState.READY


def _wait_first_ready(
    process: BaseProcess, status_path: Path, owned: dict[int, int], *, timeout_seconds: float
) -> tuple[int, int, float]:
    if process.pid is None:
        raise RuntimeError("isolated runner did not start")
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        if not process.is_alive():
            raise RuntimeError(f"isolated runner exited during preparation: {process.exitcode}")
        _remember_descendants(process.pid, owned)
        if _ready(status_path) and (gpu_pid := _gpu_child(process.pid, owned)) is not None:
            identity = _process_identity(gpu_pid)
            if identity is not None:
                return gpu_pid, identity[1], time.monotonic() - started
        time.sleep(0.1)
    raise TimeoutError(f"isolated runner did not prepare within {timeout_seconds:g} seconds")


def _signal_owned(pid: int, started: int, signal_number: int) -> None:
    identity = _process_identity(pid)
    if identity is None or identity[0] == "Z" or identity[1] != started:
        raise RuntimeError("GPU child identity changed before fault injection")
    os.kill(pid, signal_number)


def _wait_recovered(
    process: BaseProcess,
    status_path: Path,
    old_pid: int,
    old_started: int,
    owned: dict[int, int],
    injected_at: float,
) -> tuple[int, int, float, float]:
    if process.pid is None:
        raise RuntimeError("isolated runner lost its process ID")
    old_terminated_at: float | None = None
    while time.monotonic() - injected_at < 120:
        if not process.is_alive():
            raise RuntimeError(f"isolated runner exited instead of recovering: {process.exitcode}")
        _remember_descendants(process.pid, owned)
        if old_terminated_at is None and old_pid not in _live_owned({old_pid: old_started}):
            old_terminated_at = time.monotonic()
        if old_terminated_at is None and _gpu_child(process.pid, owned) not in (None, old_pid):
            raise RuntimeError("replacement GPU process started before the old model exited")
        if old_terminated_at is not None and _ready(status_path):
            new_pid = _gpu_child(process.pid, owned)
            if new_pid is not None and new_pid != old_pid:
                identity = _process_identity(new_pid)
                if identity is not None:
                    return new_pid, identity[1], old_terminated_at - injected_at, time.monotonic() - injected_at
        time.sleep(0.1)
    raise TimeoutError("isolated runner did not recover within 120 seconds")


def _wait_terminal(process: BaseProcess, status_path: Path, owned: dict[int, int], injected_at: float) -> float:
    if process.pid is None:
        raise RuntimeError("isolated runner lost its process ID")
    while time.monotonic() - injected_at < 5:
        _remember_descendants(process.pid, owned)
        if not process.is_alive():
            process.join(timeout=0)
            if status_path.exists():
                raise RuntimeError("failed runner still advertises readiness")
            return time.monotonic() - injected_at
        time.sleep(0.05)
    raise TimeoutError("failed runner remained alive after its second inference-process failure")


def _cleanup(process: BaseProcess, owned: dict[int, int]) -> Cleanup:
    started = time.monotonic()
    pid = process.pid
    if pid is not None:
        _remember_descendants(pid, owned)
    forced_runner = False
    if process.is_alive() and pid is not None:
        with suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)
        process.join(timeout=4)
    if process.is_alive():
        forced_runner = True
        process.kill()
        process.join(timeout=1)
    elif pid is not None:
        process.join(timeout=0)
    forced = _live_owned(owned)
    for child_pid in forced:
        with suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
    deadline = time.monotonic() + 1
    remaining = _live_owned(owned)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.02)
        remaining = _live_owned(owned)
    return Cleanup(time.monotonic() - started, forced_runner, forced, remaining)


def qualify_idle_runner_faults(output: Path, bundle: Path, account: Path, *, slippi_port: int) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"immutable fault report already exists: {output}")
    if not bundle.is_file() or not account.is_file():
        raise FileNotFoundError("fault qualification requires the bundle and unused account file")
    output.mkdir(parents=True)
    source_hashes = _source_hashes()
    git_sha = source_git_sha()
    manifest = {
        "schema_version": 1,
        "workload": "one idle compiled netplay runner; no reservation, Dolphin, or Slippi login",
        "started_at": datetime.now(UTC).isoformat(),
        "source_git_sha": git_sha,
        "source_file_sha256": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "bundle_sha256": _sha256(bundle),
        "account_sha256": _sha256(account),
        "slippi_port": slippi_port,
        "uv_lock_sha256": _sha256(Path(__file__).resolve().parents[3] / "uv.lock"),
    }
    _write_json(output / "manifest.json", manifest)
    status_path = output / "runner-status.json"
    resources = ExitStack()
    worker_url = resources.enter_context(local_worker(output / "worker-state", port=free_port()))
    admin = AdminClient(QueueEndpoint(worker_url, DEV_ADMIN_TOKEN))
    resources.callback(admin.close)
    policy = policy_config_for(bundle)
    admin.put_policy(policy)
    account_digest = sha256_file(account)
    admin.put_accounts((Account(account_connect_code(account), account_key(account_digest), account_digest),))
    endpoint = QueueEndpoint(worker_url, DEV_RUNNER_TOKEN)
    sessions = RunnerClient(endpoint)
    resources.callback(sessions.close)
    session_id = new_session_id()
    started = sessions.start_session(
        session_id=session_id,
        host="idle-fault-qualification",
        bundle_sha256=policy.bundle_sha256,
        git_sha=git_sha,
        slots=1,
        wants_stream=False,
    )

    def end_session() -> None:
        with suppress(Exception):
            sessions.end_session(session_id)

    resources.callback(end_session)
    resources.enter_context(SessionReporter(sessions, session_id, SessionStatus(status_path, policy.bundle_sha256, 1)))
    config = RunnerConfig(
        queue_endpoint=endpoint,
        session_id=started.session_id,
        policy=bundle,
        user_jsons=(account,),
        slippi_ports=(slippi_port,),
        iso_path=Path(ISO_PATH),
        dolphin_path=Path(NETPLAY_EMULATOR_PATH),
        replay_dir=output / "replays",
        status_path=status_path,
        git_sha=git_sha,
        device="cuda",
        seed=0,
        compiled=True,
        batch_wait_seconds=0.0005,
        measurement_dir=output / "match-measurements",
        publish_replays=False,
    )
    process = mp.get_context("spawn").Process(target=run, args=(config,), name="hal-059-idle-fault-qualification")
    owned: dict[int, int] = {}
    measured: dict[str, object] = {}
    failure: str | None = None
    primary_error: BaseException | None = None
    try:
        process.start()
        first_pid, first_started, measured["startup_seconds"] = _wait_first_ready(
            process, status_path, owned, timeout_seconds=config.preparation_timeout_seconds + 30
        )
        measured["first_gpu_pid"] = first_pid
        measured["first_gpu_start_ticks"] = first_started
        first_injected_at = time.monotonic()
        _signal_owned(first_pid, first_started, signal.SIGSTOP)
        measured["first_fault"] = "SIGSTOP to own GPU child"
        second_pid, second_started, old_exit_seconds, recovery_seconds = _wait_recovered(
            process, status_path, first_pid, first_started, owned, first_injected_at
        )
        measured["old_gpu_termination_seconds"] = old_exit_seconds
        measured["recovery_ready_seconds"] = recovery_seconds
        measured["second_gpu_pid"] = second_pid
        measured["second_gpu_start_ticks"] = second_started
        if old_exit_seconds > 3.5 or recovery_seconds > 120:
            raise AssertionError("idle inference hang violated termination or recovery bound")
        second_injected_at = time.monotonic()
        _signal_owned(second_pid, second_started, signal.SIGKILL)
        measured["second_fault"] = "SIGKILL to replacement GPU child"
        measured["terminal_seconds"] = _wait_terminal(process, status_path, owned, second_injected_at)
        if process.exitcode in (None, 0) or float(measured["terminal_seconds"]) > 4:
            raise AssertionError("runner failed to become unavailable after its second engine failure")
        measured["reservation_count"] = sum(
            event.get("job_id") is not None for event in admin.events(job=None, session=None, since=None)
        )
        if measured["reservation_count"] != 0:
            raise AssertionError("idle failure fixture unexpectedly created a reservation")
    except BaseException as error:
        primary_error = error
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        cleanup = _cleanup(process, owned)
        resources.close()
        unchanged = _source_hashes() == source_hashes
        if cleanup.remaining_descendants or not unchanged:
            issue = RuntimeError("idle runner left a process alive or source files changed during qualification")
            if primary_error is None:
                failure = f"{type(issue).__name__}: {issue}"
            else:
                primary_error.add_note(str(issue))
        report: dict[str, object] = {
            "schema_version": 1,
            "ended_at": datetime.now(UTC).isoformat(),
            "measurements": measured,
            "failure": failure,
            "runner_exitcode": process.exitcode,
            "status_file_exists": status_path.exists(),
            "source_files_unchanged": unchanged,
            "cleanup": asdict(cleanup),
        }
        _write_json(output / "report.json", report)
        if (cleanup.remaining_descendants or not unchanged) and primary_error is None:
            raise issue
    return report
