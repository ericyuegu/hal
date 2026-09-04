"""Exec a Linux child that receives SIGKILL when its parent exits."""

import ctypes
import os
import signal
import sys
from typing import Never

_PR_SET_PDEATHSIG = 1


def main() -> Never:
    if len(sys.argv) < 2:
        raise SystemExit("pdeathsig_exec requires a command")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Close the race where the parent dies after starting this wrapper but
    # before prctl installs the signal.
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    command = sys.argv[1:]
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
