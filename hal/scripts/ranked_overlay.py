"""Reload the Ranked value overlay without restarting Dolphin or inference."""

import signal
import threading
from pathlib import Path

import tyro

from hal.netplay_service.value_meter import run_overlay


def main(run_dir: Path) -> None:
    stop = threading.Event()

    def request_stop(_number: int, _frame: object) -> None:
        stop.set()

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, request_stop)
    run_overlay(run_dir, stop)


if __name__ == "__main__":
    tyro.cli(main)
