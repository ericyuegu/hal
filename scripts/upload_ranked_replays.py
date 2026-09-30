"""Upload completed ranked runs once, or watch for new game records."""

import signal
import threading
from pathlib import Path

import tyro

from hal.eval.ranked_replays import RankedReplayUploads
from hal.eval.ranked_replays import upload_pending


def main(root: Path, watch: bool = False) -> None:
    if not root.is_dir():
        raise ValueError(f"ranked output directory does not exist: {root}")
    if not watch:
        print(f"Uploaded {upload_pending(root)} ranked replays")
        return
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())
    signal.signal(signal.SIGINT, lambda *_args: stop.set())
    with RankedReplayUploads(root):
        stop.wait()


if __name__ == "__main__":
    tyro.cli(main)
