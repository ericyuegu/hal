"""Run the queue Worker under `wrangler dev` with local storage and development tokens."""

import hashlib
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx

DEV_RUNNER_TOKEN: Final[str] = "dev-runner-token"
DEV_ADMIN_TOKEN: Final[str] = "dev-admin-token"
WORKER_PROJECT: Final[Path] = Path(__file__).resolve().parents[2] / "web" / "netplay-api"


class LocalWorkerError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _tail(path: Path) -> str:
    return path.read_text(errors="replace")[-4000:]


def _wait_ready(process: subprocess.Popen[bytes], url: str, log_path: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise LocalWorkerError(f"wrangler dev exited with {code}:\n{_tail(log_path)}")
        try:
            if httpx.get(f"{url}/v1/capacity", timeout=1.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.25)
    raise LocalWorkerError(f"wrangler dev did not answer within {timeout_seconds:g} s:\n{_tail(log_path)}")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@contextmanager
def local_worker(state_dir: Path, *, port: int, startup_timeout_seconds: float = 90.0) -> Iterator[str]:
    """Yield the base URL of a local Worker whose storage lives under `state_dir`."""
    wrangler = WORKER_PROJECT / "node_modules" / ".bin" / "wrangler"
    if not wrangler.is_file():
        raise LocalWorkerError(f"{wrangler} is missing; run `npm ci` in {WORKER_PROJECT}")
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "wrangler.log"
    command = [
        str(wrangler),
        "dev",
        "--ip",
        "127.0.0.1",
        "--port",
        str(port),
        "--persist-to",
        str(state_dir / "storage"),
        "--show-interactive-dev-session=false",
        "--var",
        f"RUNNER_TOKEN_SHA256:{_digest(DEV_RUNNER_TOKEN)}",
        "--var",
        f"ADMIN_TOKEN_SHA256:{_digest(DEV_ADMIN_TOKEN)}",
    ]
    url = f"http://127.0.0.1:{port}"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=WORKER_PROJECT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(process, url, log_path, startup_timeout_seconds)
            yield url
        finally:
            _stop(process)
