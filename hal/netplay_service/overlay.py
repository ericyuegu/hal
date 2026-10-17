"""Run the Twitch overlay independently of gameplay and inference."""

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path

from loguru import logger

from hal.netplay_service.queue_client import QueueProtocolError
from hal.netplay_service.stream import StreamSupervisor
from hal.netplay_service.stream import playing_player_count
from hal.netplay_service.stream_lease import read_stream_lease


def run(status_path: Path, display: str, stop: threading.Event, *, bandwidth_test: bool = False) -> None:
    release_file = os.environ.get("HAL_OVERLAY_RELEASE_FILE")
    git_sha = Path(release_file).read_text().strip() if release_file else None
    pulse = status_path.parent / "pulse" / "native"
    with StreamSupervisor(
        display,
        status_path.with_suffix(".stream.json"),
        status_path.with_suffix(".overlay.txt"),
        lambda: playing_player_count(status_path, now=time.time()),
        {"PULSE_SERVER": f"unix:{pulse}", "PULSE_SINK": "hal_stream"},
        bandwidth_test=bandwidth_test,
        slot_status_path=status_path.with_name(f"{status_path.name}.slot-0.json"),
    ) as supervisor:
        previous_error: str | None = None
        while not stop.is_set():
            try:
                lease = read_stream_lease(status_path.with_suffix(".stream-lease.json"))
                grant = lease.current_grant(time.time())
                previous_error = None
            except (OSError, ValueError, QueueProtocolError) as error:
                grant = None
                # Lease data contains the stream key; log only the error class.
                if type(error).__name__ != previous_error:
                    previous_error = type(error).__name__
                    logger.warning("overlay waiting for a valid stream lease: {}", previous_error)
            supervisor.set_grant(grant)
            heartbeat = status_path.with_suffix(".overlay-service.json")
            temporary = heartbeat.with_suffix(".partial")
            temporary.write_text(json.dumps({"schema_version": 1, "updated_at": time.time(), "git_sha": git_sha}))
            temporary.replace(heartbeat)
            stop.wait(0.25)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--display", required=True)
    parser.add_argument("--twitch-bandwidth-test", action="store_true")
    args = parser.parse_args(argv)
    stop = threading.Event()

    def shutdown(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    run(args.status_path, args.display, stop, bandwidth_test=args.twitch_bandwidth_test)


if __name__ == "__main__":
    main()
