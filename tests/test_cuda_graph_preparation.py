"""Torch compiler setup must stay bounded to declared shape preparation."""

from functools import partial

import pytest
import torch
from torch._dynamo import config as dynamo_config

from hal.inference.cuda_graph import CapturedCall
from hal.inference.cuda_graph import _require_tested_compiler_version
from hal.inference.cuda_graph import prepared_shape_compilation


def _increment_state(state: torch.Tensor) -> torch.Tensor:
    state.add_(1)
    return state


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_capture_binds_the_declared_device() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    state = torch.zeros(1, device=device)
    call = CapturedCall(partial(_increment_state, state), (), (state,), device=device)
    assert call.device == device
    assert state.item() == 0
    assert call(()).item() == 1
    assert call(()).item() == 2

    with pytest.raises(ValueError, match="declared device"):
        CapturedCall(partial(_increment_state, torch.zeros(1)), (), (torch.zeros(1),), device=device)
    with pytest.raises(ValueError, match="needs a CUDA device"):
        CapturedCall(partial(_increment_state, state), (), (state,), device=torch.device("cpu"))
