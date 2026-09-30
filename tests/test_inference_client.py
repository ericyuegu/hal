"""Bounded, persistent delivery over the local inference pipe."""

import threading
import time
from dataclasses import replace
from multiprocessing import Pipe
from multiprocessing.connection import Connection

import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.inference.api import PolicyInput
from hal.inference.api import PolicySpec
from hal.inference.api import PredictionRequest
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import action_plan
from hal.inference.client import InferenceClient
from hal.inference.client import InferenceUnavailable
from hal.inference.client import StreamAck
from hal.inference.client import StreamAdmission

PROFILE = PreparedInferenceProfile("test", "a" * 64, "window", 4, 0, (1, 2, 4), 1)


def _request(generation: int, sequence: int) -> PredictionRequest:
    item = PolicyInput(991, sequence, 1, {}, NEUTRAL_CONTROLLER_ACTION)
    return PredictionRequest(991, generation, sequence, sequence, (item,), ())


def _client(timeout: float = 1.0) -> tuple[InferenceClient, Connection]:
    parent, child = Pipe()
    client = InferenceClient(
        PolicySpec("test", "test", (), (0,)), 8, child, threading.Event(), {0: PROFILE}, timeout_seconds=timeout
    )
    return client, parent


def _admit(client: InferenceClient, parent: Connection) -> None:
    parent.send(StreamAck(991, 1, "admitted"))
    assert client.start_match(991, 0) == 1
    assert parent.recv() == StreamAdmission(991, 1, PROFILE)


def test_one_delivery_thread_handles_repeated_requests() -> None:
    client, parent = _client()
    _admit(client, parent)
    delivered = []

    def server() -> None:
        for _ in range(3):
            request = parent.recv()
            delivered.append(request)
            parent.send(action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4, state_value=0.0))

    server_thread = threading.Thread(target=server)
    server_thread.start()
    try:
        worker_thread = None
        for sequence in range(3):
            client.submit(_request(1, sequence))
            if worker_thread is None:
                worker_thread = client._thread
            assert client._thread is worker_thread
            deadline = time.monotonic() + 1.0
            while client.poll() is None and time.monotonic() < deadline:
                time.sleep(0.001)
            assert not client.busy
        assert len(delivered) == 3
        assert worker_thread is not None and worker_thread.is_alive()
    finally:
        client.close()
        parent.close()
        server_thread.join(timeout=1.0)
    assert not server_thread.is_alive()
    assert worker_thread is not None and not worker_thread.is_alive()


def test_hung_reply_times_out_independently_of_delivery_thread() -> None:
    client, parent = _client(timeout=0.03)
    _admit(client, parent)
    try:
        client.submit(_request(1, 0))
        assert parent.poll(0.2)
        parent.recv()
        time.sleep(0.04)
        with pytest.raises(InferenceUnavailable, match="timed out"):
            client.poll()
        assert not client.busy
        with pytest.raises(InferenceUnavailable, match="timed out"):
            client.submit(_request(1, 1))
    finally:
        client.close()
        parent.close()


def test_close_reaps_delivery_thread_while_live_peer_never_replies() -> None:
    client, parent = _client(timeout=0.2)
    _admit(client, parent)
    client.submit(_request(1, 0))
    assert parent.poll(0.2)
    parent.recv()
    worker_thread = client._thread
    assert worker_thread is not None and worker_thread.is_alive()
    client.close()
    try:
        assert not worker_thread.is_alive()
    finally:
        parent.close()


def test_delivery_latency_ends_when_response_arrives_before_later_poll() -> None:
    client, parent = _client(timeout=0.1)
    _admit(client, parent)

    def reply() -> None:
        request = parent.recv()
        parent.send(action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4, state_value=0.0))

    server = threading.Thread(target=reply)
    server.start()
    try:
        client.submit(_request(1, 0))
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            with client._lock:
                received = client._response is not None
                receipt_latency = client._received_at - client._started
            if received:
                break
            time.sleep(0.001)
        else:
            pytest.fail("test server did not deliver its response")
        assert receipt_latency < 0.1
        time.sleep(0.12)
        assert client.poll() is not None
        assert client.last_latency == pytest.approx(receipt_latency, abs=0.002)
    finally:
        server.join(timeout=1.0)
        client.close()
        parent.close()


def test_response_received_after_request_deadline_is_rejected_without_early_poll() -> None:
    client, parent = _client(timeout=0.03)
    _admit(client, parent)
    try:
        client.submit(_request(1, 0))
        request = parent.recv()
        time.sleep(0.04)
        parent.send(action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4, state_value=0.0))
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            with client._lock:
                if client._response is not None:
                    break
            time.sleep(0.001)
        with pytest.raises(InferenceUnavailable, match="timed out"):
            client.poll()
        assert not client.busy
    finally:
        client.close()
        parent.close()


def test_late_admission_acknowledgement_cannot_open_a_match() -> None:
    client, parent = _client(timeout=0.03)

    def late_ack() -> None:
        assert parent.recv() == StreamAdmission(991, 1, PROFILE)
        time.sleep(0.04)
        parent.send(StreamAck(991, 1, "admitted"))

    server = threading.Thread(target=late_ack)
    server.start()
    try:
        with pytest.raises(InferenceUnavailable, match="timed out"):
            client.start_match(991, 0)
        assert client.generation == 0
        assert not client.busy
    finally:
        server.join(timeout=1.0)
        client.close()
        parent.close()


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf"), 0.0, True])
def test_client_rejects_nonfinite_or_nonpositive_timeout(timeout: float) -> None:
    parent, child = Pipe()
    try:
        with pytest.raises(ValueError, match="positive context and timeout"):
            InferenceClient(
                PolicySpec("test", "test", (), (0,)),
                8,
                child,
                threading.Event(),
                {0: PROFILE},
                timeout_seconds=timeout,
            )
    finally:
        parent.close()
        child.close()


def test_client_rejects_boolean_context_size() -> None:
    parent, child = Pipe()
    try:
        with pytest.raises(ValueError, match="positive context and timeout"):
            InferenceClient(PolicySpec("test", "test", (), (0,)), True, child, threading.Event(), {0: PROFILE})
    finally:
        parent.close()
        child.close()


def test_wrong_active_response_identity_fails_stream() -> None:
    client, parent = _client()
    _admit(client, parent)
    try:
        request = _request(1, 0)
        client.submit(request)
        assert parent.poll(0.2)
        parent.recv()
        parent.send(replace(action_plan(request, (NEUTRAL_CONTROLLER_ACTION,) * 4, state_value=0.0), generation=2))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                client.poll()
            except InferenceUnavailable as error:
                assert "identity" in str(error)
                break
            time.sleep(0.001)
        else:
            pytest.fail("client did not reject the mismatched active response")
        assert not client.busy
    finally:
        client.close()
        parent.close()


def test_start_match_rejects_outstanding_request() -> None:
    client, parent = _client()
    _admit(client, parent)
    try:
        client.submit(_request(1, 0))
        with pytest.raises(RuntimeError, match="outstanding"):
            client.start_match(991, 0)
    finally:
        client.close()
        parent.close()


def test_boolean_generation_ack_does_not_admit_a_stream() -> None:
    client, parent = _client()
    parent.send(StreamAck(991, True, "admitted"))
    try:
        with pytest.raises(InferenceUnavailable, match="acknowledgement differs"):
            client.start_match(991, 0)
        assert parent.recv() == StreamAdmission(991, 1, PROFILE)
        assert not client.busy
    finally:
        client.close()
        parent.close()
