import threading
import time
from multiprocessing import Pipe

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.inference.api import PolicyInput
from hal.inference.api import PolicyOutput
from hal.inference.api import PolicySpec
from hal.inference.api import RuntimeConfig
from hal.netplay_service.inference import ContinuousBatcher
from hal.netplay_service.inference import RemotePolicy
from hal.netplay_service.inference import ServingArena


class _Policy:
    spec = PolicySpec(
        name="fake",
        backend="tests.fake",
        required_observation_fields=("integer", "floating"),
        supported_transport_delays=(2, 3),
        requires_player_identity=True,
    )

    def __init__(self) -> None:
        self.batches: list[tuple[PolicyInput, ...]] = []

    def prepare(self, _config: RuntimeConfig) -> None:
        pass

    def step(self, inputs: tuple[PolicyInput, ...]) -> tuple[PolicyOutput, ...]:
        self.batches.append(inputs)
        return tuple(
            PolicyOutput(
                item.stream_id,
                ControllerAction(float(item.stream_id) / 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
            )
            for item in inputs
        )


def _input(stream_id: int, delay: int) -> PolicyInput:
    return PolicyInput(
        stream_id=stream_id,
        frame_id=10 + stream_id,
        controlled_port=1,
        observation={"integer": 3, "floating": 1.25},
        applied_action=NEUTRAL_CONTROLLER_ACTION,
        pending_actions=(NEUTRAL_CONTROLLER_ACTION,) * delay,
        player_identity="IBDW#0",
        reset=True,
    )


def test_continuous_batcher_combines_mixed_delays_and_preserves_types() -> None:
    runtime = RuntimeConfig(2, (2, 3))
    policy = _Policy()
    parent_0, child_0 = Pipe()
    parent_1, child_1 = Pipe()
    with ServingArena.create(2, 3, policy.spec.required_observation_fields) as arena:
        stop = threading.Event()
        batcher = ContinuousBatcher(
            policy,
            runtime,
            arena,
            {0: parent_0, 1: parent_1},
            batch_wait_seconds=0.02,
        )
        server = threading.Thread(target=batcher.serve, args=(stop,), daemon=True)
        server.start()
        clients = (
            RemotePolicy(policy.spec, runtime, arena, child_0, 0),
            RemotePolicy(policy.spec, runtime, arena, child_1, 1),
        )
        results: list[tuple[PolicyOutput, ...] | None] = [None, None]

        def call(index: int) -> None:
            results[index] = clients[index].step((_input(7 + index, 2 + index),))

        threads = [threading.Thread(target=call, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1)
        stop.set()
        server.join(timeout=1)

        assert [result[0].action.main_x for result in results if result is not None] == [0.7, 0.8]
        assert len(policy.batches) == 1
        assert (batcher.batch_calls, batcher.batch_items, batcher.max_batch_items) == (1, 2, 2)
        first, second = policy.batches[0]
        assert (len(first.pending_actions), len(second.pending_actions)) == (2, 3)
        assert isinstance(first.observation["integer"], int)
        assert isinstance(first.observation["floating"], float)


def test_continuous_batcher_does_not_wait_for_an_idle_slot() -> None:
    runtime = RuntimeConfig(2, (2, 3))
    policy = _Policy()
    parent_0, child_0 = Pipe()
    parent_1, _child_1 = Pipe()
    with ServingArena.create(2, 3, policy.spec.required_observation_fields) as arena:
        stop = threading.Event()
        batcher = ContinuousBatcher(
            policy,
            runtime,
            arena,
            {0: parent_0, 1: parent_1},
            batch_wait_seconds=0.005,
        )
        server = threading.Thread(target=batcher.serve, args=(stop,), daemon=True)
        server.start()
        client = RemotePolicy(policy.spec, runtime, arena, child_0, 0)

        started = time.perf_counter()
        output = client.step((_input(1, 2),))
        elapsed = time.perf_counter() - started
        stop.set()
        server.join(timeout=1)

        assert output[0].stream_id == 1
        assert elapsed < 0.1
        assert len(policy.batches) == 1
        assert len(policy.batches[0]) == 1
