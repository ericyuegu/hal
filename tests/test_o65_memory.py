import copy
import dataclasses
import json
import weakref
from pathlib import Path
from typing import Any

import pytest
import torch
from test_065_loop import exp
from test_065_loop import trainer
from test_065_loop import transformer_trainer

from hal.inference.cuda_graph import prepared_shape_compilation


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [0, 7, 4103])
def test_chunked_factor_kl_preserves_full_reduction(dtype: torch.dtype, count: int) -> None:
    run = trainer("cpu")
    batch = run.prepare(run.engine.collect(0))
    generator = torch.Generator().manual_seed(65)
    probabilities, references = [], []
    for size in (7, 5, 3, 2):
        p = torch.randn(count, 4, size, dtype=dtype, generator=generator).softmax(-1)
        q = torch.randn(count, 4, size, dtype=dtype, generator=generator).softmax(-1)
        # Identical zero support remains legal, even at chunk boundaries.
        p[..., -1] = q[..., -1] = 0
        probabilities.append(p / p.sum(-1, keepdim=True))
        references.append(q / q.sum(-1, keepdim=True))
    executed = (torch.arange(count)[:, None] + torch.arange(16)[None]) % 3 != 0
    for group, (p, q) in enumerate(zip(probabilities, references, strict=True)):
        p.masked_fill_(~executed[:, group::4, None], torch.nan)
        q.masked_fill_(~executed[:, group::4, None], -1)
    old = dataclasses.replace(batch.old, distributions=tuple(probabilities))
    reference = dataclasses.replace(batch.old, distributions=tuple(references))
    expected = torch.stack(
        [
            exp._distribution_kl(p, q, executed[:, group::4, None])
            for group, (p, q) in enumerate(zip(probabilities, references, strict=True))
        ],
        -1,
    ).flatten(1)
    actual = exp.factor_kl(old, reference, executed)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype
    if count > 4096:
        probabilities[0][-1, 0, 0] = torch.nan
        executed[-1, 0] = True
        with pytest.raises(ValueError, match="finite"):
            exp.factor_kl(old, reference, executed)


def test_memory_inventory_counts_views_once() -> None:
    value = torch.zeros(11, 7)
    seen: set[tuple[torch.device, int]] = set()
    assert exp.tensor_storage_bytes({"base": value, "view": (value[:3],)}, seen) == value.untyped_storage().nbytes()
    assert exp.tensor_storage_bytes(value[3:], seen) == 0
    assert exp.tensor_storage_bytes(value.clone(), seen) == value.untyped_storage().nbytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_inventory_preserves_large_graph_pool_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.ones(4, device="cuda")
    source.square()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(257):
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin()
            output = source.square()
            graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    assert max(graph.pool()) > 256
    del graph, output
    run = transformer_trainer("cuda", decoder_precision="bf16")
    run.reference.prepare_tick_weights()
    run.reference.reference_forward_model = copy.deepcopy(run.reference.model)
    call = exp.CapturedCall(lambda: source.square(), (source,), device=source.device)
    run.policy.tick_graphs[((run.engine.rings[0], (0, 4)), 1, 4, 4)] = call
    run.startup_memory.output = tmp_path
    snapshot = torch.cuda.memory_snapshot
    expected = [
        segment for segment in snapshot(include_traces=False) if segment["segment_pool_id"] == call.graph.pool()
    ]

    def safe_snapshot(mempool_id: tuple[int, int] | None = None, include_traces: bool = True) -> list[dict[str, Any]]:
        # Reject the unsafe binding before it can corrupt this test process.
        assert mempool_id is None
        return snapshot(include_traces=include_traces)

    monkeypatch.setattr(torch.cuda, "memory_snapshot", safe_snapshot)
    run.write_memory_inventory("startup")
    report = json.loads((tmp_path / "memory-inventory-startup.json").read_text())
    assert report["pools"][str(call.graph.pool())] == {
        "allocated_bytes": sum(segment["allocated_size"] for segment in expected),
        "reserved_bytes": sum(segment["total_size"] for segment in expected),
    }
    assert report["unique_storage_bytes"]["policy1_reference_buffers"] > 0
    call.graph.replay()
    torch.testing.assert_close(call.output, source.square(), rtol=0, atol=0)


@pytest.mark.parametrize("extra", ["reference", "mask"])
def test_chunked_kl_rejects_unmatched_tail_rows(extra: str) -> None:
    run = trainer("cpu")
    batch = run.prepare(run.engine.collect(0))
    p = torch.full((4103, 4, 2), 0.5)
    q = torch.full((4104 if extra == "reference" else 4103, 4, 2), 0.5)
    mask = torch.ones(4104 if extra == "mask" else 4103, 16, dtype=torch.bool)
    old = dataclasses.replace(batch.old, distributions=(p,) * 4)
    reference = dataclasses.replace(batch.old, distributions=(q,) * 4)
    with pytest.raises(ValueError, match="matching decision rows"):
        exp.factor_kl(old, reference, mask)


def test_kl_support_error_locates_factor_chunk_and_probability() -> None:
    run = trainer("cpu")
    batch = run.prepare(run.engine.collect(0))
    p = torch.full((4103, 4, 2), 0.5)
    q = p.clone()
    q[-1, 0] = torch.tensor([0.0, 1.0])
    old = dataclasses.replace(batch.old, distributions=(p,) * 4)
    reference = dataclasses.replace(batch.old, distributions=(p, p, q, p))
    with pytest.raises(ValueError) as failure:
        exp.factor_kl(old, reference, torch.ones(4103, 16, dtype=torch.bool))
    message = str(failure.value)
    assert "KL factor 2, decisions [4096, 4103)" in message
    assert "identical support; flat category 48, shape (7, 4, 2)" in message
    assert "policy=-0.693" in message and "reference=-inf" in message


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_releases_unused_backward_graphs_without_changing_arithmetic(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    run = transformer_trainer(device, decoder_precision="bf16" if device == "cuda" else "fp32")
    windows = run.prepare(run.engine.collect(0)).windows
    reference = run.reference
    original_history, original_replay = reference.window_history, reference.window_replay

    def forward(retain: bool) -> exp.ReplayOutput:
        histories: list[torch.Tensor] = []
        outputs: list[exp.ReplayOutput] = []
        history_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        output_refs: list[weakref.ReferenceType[torch.Tensor]] = []

        def history(*args: Any) -> torch.Tensor:
            value = original_history(*args)
            assert value.requires_grad and value.grad_fn is not None
            if retain:
                histories.append(value)
            else:
                history_refs.append(weakref.ref(value))
            return value

        def replay(model: Any, hidden: torch.Tensor, *args: Any, grad: bool) -> exp.ReplayOutput:
            assert grad and hidden.requires_grad and hidden.is_leaf
            if retain:
                # Control: preserve the original non-leaf input and all backward
                # activations until the complete reference forward returns.
                hidden = histories[-1]
            else:
                assert all(saved() is None for saved in history_refs)
                assert all(saved() is None for saved in output_refs)
            value = original_replay(model, hidden, *args, grad=grad)
            if retain:
                outputs.append(value)
            else:
                output_refs.append(weakref.ref(value.log_probs_WKF))
            return value

        with monkeypatch.context() as patch, torch.no_grad(), prepared_shape_compilation(64):
            patch.setattr(reference, "window_history", history)
            patch.setattr(reference, "window_replay", replay)
            result = reference.reference_forward(windows.inputs(), len(windows.rows), 16)
        assert not result.log_probs_WKF.requires_grad
        assert all(not p.requires_grad for p in result.distributions)
        return result

    expected, actual = forward(True), forward(False)
    torch.testing.assert_close(actual.log_probs_WKF, expected.log_probs_WKF, rtol=0, atol=0)
    for got, saved in zip(actual.distributions, expected.distributions, strict=True):
        torch.testing.assert_close(got, saved, rtol=0, atol=0)
    for field in dataclasses.fields(actual.features):
        torch.testing.assert_close(
            getattr(actual.features, field.name), getattr(expected.features, field.name), rtol=0, atol=0
        )
