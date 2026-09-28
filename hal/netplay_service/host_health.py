"""HTTP health check for a managed netplay host."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Final

from hal.netplay_service.health import read_runner_status

DEFAULT_MAX_AGE_SECONDS: Final[float] = 10.0


def runner_is_healthy(status_path: Path, now: float, max_age: float = DEFAULT_MAX_AGE_SECONDS) -> bool:
    if max_age <= 0:
        raise ValueError("max_age must be positive")
    try:
        status = read_runner_status(status_path)
    except ValueError:
        return False
    age = now - status.updated_at
    return 0 <= age <= max_age and status.healthy_slots > 0


def _handler(status_path: Path, max_age: float, clock: Callable[[], float]) -> type[BaseHTTPRequestHandler]:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            healthy = runner_is_healthy(status_path, clock(), max_age)
            body = b"ok\n" if healthy else b"unhealthy\n"
            self.send_response(HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    return HealthHandler


def serve(
    status_path: Path,
    host: str,
    port: int,
    max_age: float = DEFAULT_MAX_AGE_SECONDS,
    *,
    clock: Callable[[], float] = time.time,
) -> None:
    server = ThreadingHTTPServer((host, port), _handler(status_path, max_age, clock))
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9101)
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_SECONDS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("--port must be between 1 and 65535")
    if args.max_age <= 0:
        parser.error("--max-age must be positive")
    serve(args.status_path, args.host, args.port, args.max_age)


if __name__ == "__main__":
    main()
