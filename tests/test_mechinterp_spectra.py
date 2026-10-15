import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "spectra.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_spectra", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
spectra = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = spectra
_SPEC.loader.exec_module(spectra)


def test_spectra_distinguish_rank_and_scale() -> None:
    identity = torch.eye(4)
    baseline = spectra.matrix_metrics(identity)
    scaled = spectra.matrix_metrics(3 * identity)
    rank_one = spectra.matrix_metrics(torch.diag(torch.tensor([2.0, 0.0, 0.0, 0.0])))
    assert baseline["stable_rank"] == pytest.approx(4)
    assert baseline["effective_rank"] == pytest.approx(4)
    assert scaled["frobenius"] == pytest.approx(6)
    assert scaled["stable_rank"] == baseline["stable_rank"]
    assert rank_one["stable_rank"] == pytest.approx(1)
    assert rank_one["top1_energy"] == pytest.approx(1)


def test_matrix_change_separates_rotation_from_scaling() -> None:
    before = torch.tensor([[1.0, 0.0]])
    scaled = spectra.matrix_change(before, 2 * before)
    rotated = spectra.matrix_change(before, torch.tensor([[0.0, 1.0]]))
    assert scaled["angle_degrees"] == pytest.approx(0)
    assert scaled["norm_ratio"] == pytest.approx(2)
    assert rotated["angle_degrees"] == pytest.approx(90)
    assert rotated["relative_difference"] == pytest.approx(math.sqrt(2))


def test_logical_qkv_splits_preserve_identity() -> None:
    fused = torch.arange(12.0).reshape(6, 2)
    matrices = spectra.logical_matrices({"blocks.0.attn.c_attn.weight": fused})
    assert len(matrices) == 4
    assert torch.equal(matrices["blocks.0.attn.c_attn.weight/q"], fused[:2])
    assert torch.equal(matrices["blocks.0.attn.c_attn.weight/k"], fused[2:4])
    with pytest.raises(ValueError, match="unsupported fused QKV"):
        spectra.logical_matrices({"blocks.0.attn.c_attn.weight": torch.zeros(5, 2)})


def test_checkpoint_state_rejects_wrong_format(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"unrelated": torch.ones(2)}, path)
    with pytest.raises(ValueError, match="expected a HAL checkpoint"):
        spectra.checkpoint_state(path)


def test_checkpoint_state_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    expected = torch.arange(6).reshape(2, 3)
    torch.save({"model": {"weight": expected}, "step": 17, "cfg": {"seed": 0}}, path)
    state, metadata = spectra.checkpoint_state(path)
    assert torch.equal(state["weight"], expected)
    assert metadata["step"] == 17
