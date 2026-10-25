from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch._dynamo.utils import counters
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.utils import fresh_cache
from torch._inductor.utils import run_and_get_code
from torch.compiler import config as compiler_config

from hal.inference import inductor_compat


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(TritonScheduling, "kernel_type", TritonKernel)
    monkeypatch.setattr(compiler_config, "cache_key_tag", "")
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.mark.usefixtures("generator")
@pytest.mark.parametrize("dependency", ["torch", "triton"])
def test_barrier_rejects_untested_dependencies(monkeypatch: pytest.MonkeyPatch, dependency: str) -> None:
    if dependency == "torch":
        monkeypatch.setattr(torch, "__version__", "2.12.0")
    else:
        monkeypatch.setattr(inductor_compat.importlib.metadata, "version", lambda name: "3.7.0")
    with pytest.raises(RuntimeError, match="requires tested"):
        inductor_compat.install_reduction_barrier()
    assert TritonScheduling.kernel_type is TritonKernel
    assert compiler_config.cache_key_tag == ""


@pytest.mark.usefixtures("generator")
def test_barrier_installation_is_idempotent_and_detects_a_changed_cache_tag() -> None:
    inductor_compat.install_reduction_barrier()
    installed = TritonScheduling.kernel_type
    inductor_compat.install_reduction_barrier()
    assert TritonScheduling.kernel_type is installed
    assert compiler_config.cache_key_tag == inductor_compat.CACHE_KEY_TAG
    compiler_config.cache_key_tag = "other"
    with pytest.raises(RuntimeError, match="changed after installation"):
        inductor_compat.install_reduction_barrier()


@pytest.mark.usefixtures("generator")
@pytest.mark.parametrize("conflict", ["generator", "cache_tag"])
def test_barrier_does_not_overwrite_other_compiler_customizations(conflict: str) -> None:
    if conflict == "generator":
        TritonScheduling.kernel_type = object
    else:
        compiler_config.cache_key_tag = "other"
    with pytest.raises(RuntimeError, match="another Triton generator or compiler cache tag"):
        inductor_compat.install_reduction_barrier()


def action_head_log_probs(temporal: Tensor, temporal_bias: Tensor, trunk: Tensor, trunk_bias: Tensor) -> Tensor:
    # Retain the strided BF16 inputs and repeated trunk rows from the failing
    # actor. The centered reduction triggers the cross-thread scratch read-back.
    logits = (temporal[:, :65].float() + temporal_bias) + (trunk[:, :65].float() + trunk_bias).repeat_interleave(
        4, dim=0
    )
    return (logits - logits.mean(-1, keepdim=True)).log_softmax(-1)


@pytest.mark.usefixtures("generator")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton reduction regression requires CUDA")
def test_barrier_normalization_gradients_and_cold_warm_compiler_caches(tmp_path: Path) -> None:
    torch.manual_seed(65)
    inputs = (
        torch.randn(16384, 72, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(65, device="cuda", requires_grad=True),
        torch.randn(4096, 72, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(65, device="cuda", requires_grad=True),
    )
    exact_inputs = tuple(value.detach().double().requires_grad_(True) for value in inputs)
    temporal, temporal_bias, trunk, trunk_bias = exact_inputs
    # Independent oracle omits centering, which cannot change a softmax.
    exact_logits = temporal[:, :65] + temporal_bias + (trunk[:, :65] + trunk_bias).repeat_interleave(4, dim=0)
    expected = exact_logits.log_softmax(-1)
    cotangent = torch.randn_like(expected, dtype=torch.float32)
    expected_gradients = torch.autograd.grad((expected * cotangent.double()).sum(), exact_inputs)
    options = {"triton.cudagraphs": False, "emulate_precision_casts": True, "triton.persistent_reductions": False}
    with fresh_cache(dir=str(tmp_path)):
        unpatched = torch.compile(action_head_log_probs, fullgraph=True, options=options)
        _, sources = run_and_get_code(unpatched, *inputs)
        assert sources and not any("tl.debug_barrier()" in source for source in sources)
        torch.compiler.reset()
        inductor_compat.install_reduction_barrier()
        for warm in (False, True):
            hits_before = counters["inductor"]["fxgraph_cache_hit"]
            compiled = torch.compile(action_head_log_probs, fullgraph=True, options=options)
            actual, sources = run_and_get_code(compiled, *inputs)
            assert sources and any("tl.debug_barrier()" in source for source in sources)
            if warm:
                assert counters["inductor"]["fxgraph_cache_hit"] > hits_before
            for _ in range(10):
                actual = compiled(*inputs)
                torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=3e-6)
                log_mass = actual.double().logsumexp(-1)
                torch.testing.assert_close(log_mass, torch.zeros_like(log_mass), rtol=0, atol=1e-5)
            gradients = torch.autograd.grad((actual * cotangent).sum(), inputs)
            for gradient, exact in zip(gradients, expected_gradients, strict=True):
                # BF16 leaf gradients have a final cast; bias gradients stay FP32.
                rtol, atol = (8e-3, 1e-5) if gradient.dtype == torch.bfloat16 else (1e-5, 1e-4)
                torch.testing.assert_close(gradient.double(), exact, rtol=rtol, atol=atol)
            torch.compiler.reset()
