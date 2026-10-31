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
from hal.inference.cuda_graph import CapturePool
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


def test_capture_pool_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="pools need a CUDA device"):
        CapturePool(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_pool_prepares_all_operations_before_capture() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    pool = CapturePool(device)
    values = torch.ones(256, device=device)
    phases: list[tuple[str, bool]] = []

    def operation() -> torch.Tensor:
        output = values.sin()
        capturing = torch.cuda.is_current_stream_capturing()
        phases.append((pool.phase, capturing))
        if not capturing:
            assert all(tuple(segment["segment_pool_id"]) != pool.handle for segment in torch.cuda.memory_snapshot())
        return output

    call = CapturedCall(operation, (values,), device=device, pool=pool)
    other = CapturedCall(operation, (values,), device=device, pool=pool)
    assert phases == [("preparing", False)] * 4
    pool.capture_all()
    assert phases == [("preparing", False)] * 4 + [("capturing", True)] * 2
    assert pool.phase == "ready" and not pool.pending
    assert call.graph.pool() == other.graph.pool() == pool.handle
    torch.testing.assert_close(call((values,)), values.sin(), rtol=0, atol=0)
    with pytest.raises(ValueError, match="already captured"):
        pool.capture_all()
    with pytest.raises(ValueError, match="closed to new operations"):
        CapturedCall(operation, (values,), device=device, pool=pool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_pool_preserves_gradients_and_copied_outputs_in_either_order() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    pool = CapturePool(device)
    weight = torch.nn.Parameter(torch.arange(1, 257, device=device, dtype=torch.float32))
    values = torch.ones_like(weight)
    gradient = torch.zeros_like(weight)
    weight.grad = gradient

    def backward() -> torch.Tensor:
        gradient.zero_()
        with torch.enable_grad():
            loss = (weight * values).square().sum()
            loss.backward()
        return loss.detach()

    actor = CapturedCall(backward, (values,), (gradient,), device=device, pool=pool)
    reference = CapturedCall(lambda: (values + 3).square().sum(), (values,), device=device, pool=pool)
    pool.capture_all()
    assert actor.graph.pool() == reference.graph.pool() == pool.handle
    assert torch.count_nonzero(gradient) == 0
    retained: list[torch.Tensor] = []
    expected: list[torch.Tensor] = []
    for index, call in enumerate((reference, actor, actor, reference, actor, reference), start=1):
        inputs = torch.full_like(values, index)
        retained.append(call((inputs,)).clone())
        if call is actor:
            expected.append((weight.detach() * inputs).square().sum())
            torch.testing.assert_close(gradient, 2 * weight * inputs.square(), rtol=0, atol=0)
        else:
            expected.append((inputs + 3).square().sum())
    for actual, target in zip(retained, expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)

    other_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(other_stream), pytest.raises(ValueError, match="pool's stream"):
        actor((values,))
    other_stream.synchronize()
    with torch.cuda.stream(other_stream), pytest.raises(ValueError, match="pool's stream"):
        CapturedCall(lambda: values + 1, (values,), device=device, pool=pool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_pool_discards_preparation_after_partial_capture() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    pool = CapturePool(device)
    values = torch.ones(256, device=device)
    first = CapturedCall(lambda: values + 1, (values,), device=device, pool=pool)

    def fail_capture() -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("capture failed")
        return values + 2

    second = CapturedCall(fail_capture, (values,), device=device, pool=pool)
    third = CapturedCall(lambda: values + 3, (values,), device=device, pool=pool)
    with pytest.raises(RuntimeError, match="capture failed"):
        pool.capture_all()
    assert pool.phase == "failed" and not pool.pending
    assert first.operation is second.operation is third.operation is None
    with pytest.raises(RuntimeError, match="pool capture failed"):
        first((values,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_shared_pool_preserves_preparation_context_without_recompiling(grad_enabled: bool) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.ones(16, 16, device=device)
    weight = torch.nn.Parameter(torch.ones_like(values))
    pool = CapturePool(device)
    operation = torch.compile(lambda: values @ weight, fullgraph=True, options={"triton.cudagraphs": False})
    with (
        torch.set_grad_enabled(grad_enabled),
        torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
    ):
        call = CapturedCall(operation, (values,), device=device, pool=pool)
    with count_compilation_starts() as counter:
        assert call((values,)).dtype == torch.bfloat16
        assert counter.snapshot() == 0
        pool.capture_all()
        assert counter.snapshot() == 0
    with torch.no_grad():
        values.fill_(2)
        weight.fill_(2)
    actual = call((values,))
    assert actual.requires_grad == grad_enabled
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, torch.full_like(actual, 64), rtol=0, atol=0)
