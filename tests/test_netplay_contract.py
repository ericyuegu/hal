from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_contract import RunnerQueue


def test_remote_queue_is_a_runner_queue() -> None:
    remote = RemoteQueue(QueueEndpoint("http://127.0.0.1:8787", "t"), "sess")
    queue: RunnerQueue = remote
    assert queue is remote
    remote.close()
