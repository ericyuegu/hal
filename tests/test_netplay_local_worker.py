import os
import subprocess
import time

from hal.netplay_service.local_worker import stop_process_group


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_stop_kills_group_members_after_the_leader_exits() -> None:
    process = subprocess.Popen(
        ["sh", "-c", "sleep 60 & echo $!"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    child = int(process.stdout.readline())
    process.stdout.close()
    process.wait(timeout=5)
    assert _alive(child)

    stop_process_group(process)

    deadline = time.monotonic() + 5
    while _alive(child):
        assert time.monotonic() < deadline, f"group member {child} survived"
        time.sleep(0.05)
