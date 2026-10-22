"""One native batch and one CPU-bound thread per simulator shard."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from collections.abc import Sequence
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray

from hal.sim import native
from hal.sim.inputs import native_controller_readback
from hal.wire import ACTION_DIM

STAGING_DTYPE = np.dtype([("records", native.FRAME_DTYPE), ("observed", np.float32, (2, ACTION_DIM))])


@dataclass(frozen=True, slots=True)
class PackedStep:
    frames: int
    started: float
    wall: float
    simulator: float
    packing: float
    input_wait: float
    other: float


def _bind_cpu(cpu: int) -> None:
    os.sched_setaffinity(0, {cpu})


class NativeShard:
    """Run a batch and its buffers on a CPU thread.

    Retain inputs until the future completes. Read output views before the next
    write. Pass the previous GPU copy event's synchronize method as before_write
    to protect its source buffer. Select a CPU from the GPU's NUMA socket.
    """

    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        *,
        cpu: int,
        records: NDArray | None = None,
        reward_weights: native.RewardWeights = native.DEFAULT_REWARD_WEIGHTS,
    ) -> None:
        if type(cpu) is not int or cpu not in os.sched_getaffinity(0):
            raise ValueError("cpu must be in the caller's allowed CPU set")
        self._executor = ThreadPoolExecutor(max_workers=1, initializer=_bind_cpu, initargs=(cpu,))
        self._closed = False
        self.last_step_seconds = 0.0
        try:
            self._batch = self._executor.submit(
                native.NativeRolloutBatch, data_dir, batch_size, records=records, reward_weights=reward_weights
            ).result()
        except BaseException:
            self._executor.shutdown(wait=True)
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("NativeShard is closed")

    @property
    def library_sha256(self) -> str:
        self._check_open()
        return self._batch.library_sha256

    def reset(
        self,
        configs: Sequence[native.NativeMatch],
        mask: NDArray[np.bool_],
        *,
        before_write: Callable[[], None] | None = None,
    ) -> Future[native.NativeFrameBatch]:
        self._check_open()
        return self._executor.submit(self._reset, configs, mask, before_write)

    def _reset(
        self, configs: Sequence[native.NativeMatch], mask: NDArray[np.bool_], before_write: Callable[[], None] | None
    ) -> native.NativeFrameBatch:
        if before_write is not None:
            before_write()
        return self._batch.reset(configs, mask)

    def step(
        self,
        actions: NDArray[np.float32],
        mask: NDArray[np.bool_],
        *,
        before_write: Callable[[], None] | None = None,
    ) -> Future[native.NativeFrameBatch]:
        self._check_open()
        return self._executor.submit(self._step, actions, mask, before_write)

    def _step(
        self, actions: NDArray[np.float32], mask: NDArray[np.bool_], before_write: Callable[[], None] | None
    ) -> native.NativeFrameBatch:
        if before_write is not None:
            before_write()
        started = time.perf_counter()
        result = self._batch.step(actions, mask)
        self.last_step_seconds = time.perf_counter() - started
        return result

    def step_packed(
        self,
        actions: NDArray[np.float32],
        mask: NDArray[np.bool_],
        output: NDArray,
        ready: Future[None],
    ) -> Future[PackedStep]:
        """Borrow disjoint host slices until completion; never access a GPU.

        The caller signals readiness after its input copy completes and protects
        output from any earlier asynchronous reader before submission. An end
        stops that lane immediately; other lanes can finish the two-frame tick.
        """
        self._check_open()
        count = self._batch.batch_size
        if (
            actions.dtype != np.float32
            or actions.ndim != 4
            or not 1 <= len(actions) <= 2
            or actions.shape[1:] != (count, 2, ACTION_DIM)
            or mask.dtype != np.bool_
            or mask.shape != (count,)
            or output.dtype != STAGING_DTYPE
            or output.shape != (len(actions), count)
            or not output.flags.writeable
        ):
            raise ValueError("packed step requires FP32 host inputs, bool lanes and matching staging records")
        return self._executor.submit(self._step_packed, actions, mask.copy(), output, ready)

    def _step_packed(
        self, actions: NDArray[np.float32], active: NDArray[np.bool_], output: NDArray, ready: Future[None]
    ) -> PackedStep:
        started = time.perf_counter()
        ready.result()
        input_wait = time.perf_counter() - started
        simulator = packing = 0.0
        count = 0
        for index, inputs in enumerate(actions):
            if not active.any():
                break
            before = time.perf_counter()
            frame = self._batch.step(inputs, active)
            simulator += time.perf_counter() - before
            before = time.perf_counter()
            if frame.records is None:
                raise ValueError("packed stepping requires native records")
            np.copyto(output[index]["records"], frame.records)
            np.copyto(output[index]["observed"], native_controller_readback(frame.applied_action))
            active &= ~(frame.terminated | frame.truncated)
            packing += time.perf_counter() - before
            count += 1
        wall = time.perf_counter() - started
        return PackedStep(
            count, started, wall, simulator, packing, input_wait, wall - simulator - packing - input_wait
        )

    def save(self, lane: int) -> Future[bytes]:
        self._check_open()
        return self._executor.submit(self._batch.save, lane)

    def save_many(self, lanes: Sequence[int]) -> Future[tuple[bytes, ...]]:
        self._check_open()
        return self._executor.submit(self._save_many, tuple(lanes))

    def _save_many(self, lanes: tuple[int, ...]) -> tuple[bytes, ...]:
        if len(set(lanes)) != len(lanes):
            raise ValueError("save needs distinct lanes")
        return tuple(self._batch.save(lane) for lane in lanes)

    def capture_many(self, lanes: Sequence[int]) -> Future[tuple[native.NativeSnapshot, ...]]:
        self._check_open()
        return self._executor.submit(self._capture_many, tuple(lanes))

    def _capture_many(self, lanes: tuple[int, ...]) -> tuple[native.NativeSnapshot, ...]:
        if len(set(lanes)) != len(lanes):
            raise ValueError("capture needs distinct lanes")
        return tuple(self._batch.capture(lane) for lane in lanes)

    def restore_many(
        self,
        lanes: Sequence[int],
        states: Sequence[bytes],
        *,
        before_write: Callable[[], None] | None = None,
    ) -> Future[native.NativeFrameBatch]:
        self._check_open()
        return self._executor.submit(self._restore_many, lanes, states, before_write)

    def _restore_many(
        self, lanes: Sequence[int], states: Sequence[bytes], before_write: Callable[[], None] | None
    ) -> native.NativeFrameBatch:
        if before_write is not None:
            before_write()
        return self._batch.restore_many(lanes, states)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._executor.submit(self._batch.close).result()
            finally:
                self._executor.shutdown(wait=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()
