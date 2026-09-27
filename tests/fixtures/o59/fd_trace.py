"""Read-only per-test descriptor trace for the 059 repository check."""

import collections
import os

import pytest

_TRACE = os.open(os.environ["HAL_FD_TRACE_FILE"], os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)


def _fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


def _fd_types() -> str:
    counts: collections.Counter[str] = collections.Counter()
    try:
        names = os.listdir("/proc/self/fd")
    except OSError:
        return "unavailable"
    for name in names:
        try:
            target = os.readlink(f"/proc/self/fd/{name}")
        except OSError:
            continue
        if target.startswith("/dev/shm/"):
            category = "shm"
        elif target.startswith("socket:"):
            category = "socket"
        elif target.startswith("pipe:"):
            category = "pipe"
        elif target.startswith("anon_inode:"):
            category = "anon"
        else:
            category = target.split("/", 2)[1] if target.startswith("/") else target
        counts[category] += 1
    return repr(counts.most_common(10))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):
    before = _fd_count()
    yield
    after = _fd_count()
    detail = _fd_types() if after > 100 and (after - before > 5 or after % 50 < 8) else ""
    os.write(_TRACE, f"{before}\t{after}\t{after - before}\t{item.nodeid}\t{detail}\n".encode())
