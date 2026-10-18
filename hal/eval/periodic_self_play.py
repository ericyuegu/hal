"""Non-blocking, bounded local evaluation jobs for a training process."""

import subprocess
import sys
from pathlib import Path
from typing import IO


class PeriodicEvaluator:
    """Retain a local child process and collect its completed results.

    Call poll at update boundaries to collect results or raise a child failure.
    Call finish at normal shutdown, then close to release the process and log.
    Use a separate GPU for evaluation if training exhausts device memory.
    """

    def __init__(self, root: Path, *, every: int, arguments: tuple[str, ...]) -> None:
        if type(every) is not int or every < 1:
            raise ValueError("evaluation interval must be positive")
        if any(argument.split("=", 1)[0] in ("--checkpoint", "--output", "--split") for argument in arguments):
            raise ValueError("periodic evaluation owns checkpoint, output and selection split")
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.arguments = arguments
        self.process: subprocess.Popen[bytes] | None = None
        self.log: IO[bytes] | None = None
        self.pending: Path | None = None
        self.last_update = -1

    def poll(self) -> Path | None:
        if self.process is None or self.process.poll() is None:
            return None
        code = self.process.returncode
        if self.log is not None:
            self.log.close()
            self.log = None
        pending = self.pending
        self.process = None
        self.pending = None
        if pending is None:
            raise RuntimeError("evaluation process has no output path")
        if code != 0:
            raise RuntimeError(f"evaluation exited {code}; see {pending.with_suffix('.log')}")
        result = pending / "results.json"
        if not result.is_file():
            raise RuntimeError(f"evaluation exited without publishing {result}")
        return result

    def submit(self, checkpoint: Path, *, update: int) -> bool:
        if type(update) is not int or update < 0:
            raise ValueError("update must be nonnegative")
        if self.process is not None or update % self.every or update <= self.last_update:
            return False
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        pending = self.root / f"update-{update:08d}"
        log = pending.with_suffix(".log").open("xb")
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scripts.eval_o65",
                    "--checkpoint",
                    str(checkpoint.resolve()),
                    "--output",
                    str(pending),
                    "--split",
                    "selection",
                    *self.arguments,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        except OSError:
            log.close()
            raise
        self.process, self.log, self.pending = process, log, pending
        self.last_update = update
        return True

    def close(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.pending = None

    def finish(self) -> Path | None:
        """Wait for the scheduled job at normal shutdown and propagate failures."""
        if self.process is not None:
            self.process.wait()
        return self.poll()
