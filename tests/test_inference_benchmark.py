import multiprocessing as mp
import signal
import time
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path

import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference import benchmark
from hal.inference.api import PolicyInput
from hal.inference.client import StopSignal


def _unresponsive_benchmark_process(
    bundle: Path,
    capacity: int,
    stop: StopSignal,
    connections: tuple[Connection, ...],
    status: Connection,
) -> None:
    del bundle, capacity, stop, connections
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    status.send(benchmark._BenchmarkError("TestFailure", "preparation failed"))
    while True:
        time.sleep(1)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_benchmark_rejects_invalid_preparation_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="preparation timeout must be finite and positive"):
        benchmark.measure_process_ready_pair(
            Path("unused.hal"), (), capacity=2, concurrent=True, preparation_timeout_seconds=timeout
        )


def test_benchmark_uses_declared_timeout_and_kills_unresponsive_child(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_timeouts = []
    receive = benchmark._receive_process_status

    def record_timeout(
        process: BaseProcess, connection: Connection, timeout_seconds: float
    ) -> benchmark._BenchmarkReady | benchmark._BenchmarkDone | benchmark._BenchmarkError:
        observed_timeouts.append(timeout_seconds)
        return receive(process, connection, timeout_seconds)

    monkeypatch.setattr(benchmark, "_serve_process_batch_benchmark", _unresponsive_benchmark_process)
    monkeypatch.setattr(benchmark, "_receive_process_status", record_timeout)
    frames = tuple(PolicyInput(0, frame, 1, {}, NEUTRAL_CONTROLLER_ACTION) for frame in range(8))
    children_before = set(mp.active_children())
    with pytest.raises(RuntimeError, match="preparation failed"):
        benchmark.measure_process_ready_pair(
            Path("unused.hal"),
            frames,
            capacity=2,
            concurrent=True,
            warmup_calls=1,
            measured_calls=1,
            preparation_timeout_seconds=10.0,
        )
    assert observed_timeouts == [10.0]
    assert set(mp.active_children()) == children_before
