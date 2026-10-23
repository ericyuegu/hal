"""Torch compiler setup must stay bounded to declared shape preparation."""

from contextlib import nullcontext
from functools import partial
from unittest.mock import Mock

import pytest
import torch
from torch._dynamo import callback_handler
from torch._dynamo import config as dynamo_config

from hal.inference.cuda_graph import CaptureCounter
from hal.inference.cuda_graph import CaptureCounts
from hal.inference.cuda_graph import CapturedCall
from hal.inference.cuda_graph import _require_tested_compiler_version
from hal.inference.cuda_graph import count_compilation_starts
from hal.inference.cuda_graph import prepared_shape_compilation


def _increment_state(state: torch.Tensor) -> torch.Tensor:
    state.add_(1)
    return state


def _add_one(value: torch.Tensor) -> torch.Tensor:
    return value + 1


def test_unqualified_torch_version_rejects_private_compiler_budget() -> None:
    _require_tested_compiler_version("2.11.0+cu130")
    with pytest.raises(RuntimeError, match="requires tested Torch 2.11.0"):
        _require_tested_compiler_version("2.12.0+cu130")


def test_preparation_restores_compiler_budget_after_failure() -> None:
    prior = dynamo_config.recompile_limit
    with pytest.raises(ZeroDivisionError), prepared_shape_compilation(12):
        assert dynamo_config.recompile_limit >= 48
        raise ZeroDivisionError
    assert dynamo_config.recompile_limit == prior


def test_compilation_listener_counts_first_compile_but_not_prepared_calls() -> None:
    torch._dynamo.reset()
    compiled = torch.compile(_add_one, backend="eager", fullgraph=True)
    value = torch.ones(1)
    with count_compilation_starts() as counter:
        assert counter.snapshot() == 0
        assert torch.equal(compiled(value), value + 1)
        starts = counter.snapshot()
        assert starts >= 1
        compiled(value)
        assert counter.snapshot() == starts
    with count_compilation_starts() as counter:
        compiled(value)
        assert counter.snapshot() == 0


def test_compilation_listener_unregisters_after_exception() -> None:
    with pytest.raises(ZeroDivisionError), count_compilation_starts() as counter:
        assert any(callback is counter for callback in callback_handler.start_callbacks)
        raise ZeroDivisionError
    assert all(callback is not counter for callback in callback_handler.start_callbacks)


def test_compilation_listener_rejects_silent_removal() -> None:
    with (
        pytest.raises(RuntimeError, match="removed the post-preparation compilation listener"),
        count_compilation_starts() as counter,
    ):
        callback_handler.remove_start_callback(counter)
        counter.snapshot()
    assert all(callback is not counter for callback in callback_handler.start_callbacks)


def test_failed_graph_capture_counts_attempt_without_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Mock)
    monkeypatch.setattr(torch.cuda, "Event", Mock)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=Mock()))
    monkeypatch.setattr(torch.cuda, "Stream", Mock(return_value=Mock()))
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "graph", Mock(side_effect=RuntimeError("capture failed")))
    counter = CaptureCounter()
    with pytest.raises(RuntimeError, match="capture failed"):
        CapturedCall(lambda: torch.ones(1), (), device=torch.device("cuda:0"), counter=counter)
    assert counter.snapshot() == CaptureCounts(attempts=1, completed=0)


def test_shared_pool_requires_an_explicit_capture_stream() -> None:
    with pytest.raises(ValueError, match="explicit capture stream"):
        CapturedCall(lambda: torch.ones(1), (), device=torch.device("cuda:0"), share_pool_with=Mock())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_capture_binds_the_declared_device() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    state = torch.zeros(1, device=device)
    counter = CaptureCounter()
    call = CapturedCall(partial(_increment_state, state), (), (state,), device=device, counter=counter)
    assert call.device == device
    assert counter.snapshot() == CaptureCounts(attempts=1, completed=1)
    assert call.preparation_seconds > 0 and call.capture_seconds > 0
    assert state.item() == 0
    assert call(()).item() == 1
    assert call(()).item() == 2
    assert counter.snapshot() == CaptureCounts(attempts=1, completed=1)

    with pytest.raises(ValueError, match="declared device"):
        CapturedCall(partial(_increment_state, torch.zeros(1)), (), (torch.zeros(1),), device=device)
    with pytest.raises(ValueError, match="needs a CUDA device"):
        CapturedCall(partial(_increment_state, state), (), (state,), device=torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_serial_graph_shapes_share_a_pool_and_preserve_consumed_outputs() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    values = (torch.arange(1024, device=device).float(), torch.arange(2048, device=device).float())
    first = CapturedCall(partial(torch.sin, values[0]), (values[0],), device=device, capture_stream=stream)
    second = CapturedCall(
        partial(torch.sin, values[1]), (values[1],), device=device, share_pool_with=first.graph, capture_stream=stream
    )
    calls = (first, second)
    saved = []
    for index in (1, 0, 1, 1, 0):
        value = values[index] + len(saved)
        saved.append((calls[index]((value,)).clone(), value.sin()))
    for actual, expected in saved:
        torch.testing.assert_close(actual, expected)
