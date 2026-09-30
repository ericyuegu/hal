"""Monitor Ranked game progress and OBS without changing either process."""

import argparse
import signal
import threading
from pathlib import Path

from hal.netplay_service.stream_monitor import run_monitor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", action="store_true", help="Record CPU and RSS in this PID namespace")
    args = parser.parse_args()
    stop = threading.Event()

    def request_stop(_number: int, _frame: object) -> None:
        stop.set()

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, request_stop)
    run_monitor(args.run_dir, args.output, stop, processes=args.processes)


if __name__ == "__main__":
    main()
