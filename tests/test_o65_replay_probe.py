"""Reject gradient changes hidden by a permissive FP32-relative gate."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts.o65_replay_probe import compare
    from scripts.o65_replay_probe import load_cohort
finally:
    sys.path.pop(0)


def test_gradient_gate_rejects_a_direction_change_inside_the_fp32_error_bound() -> None:
    eager = torch.tensor([1.0, 0.0])
    high = torch.tensor([0.0, 1.0])
    candidate = torch.tensor([0.7, 0.7])
    report = compare(candidate, eager, high, eager.clone(), gradient=True)
    assert report["compiled"]["relative_l2"] < 1.5 * report["eager"]["relative_l2"]
    assert not report["passed"]


def test_gradient_gate_accepts_small_error_with_the_same_direction() -> None:
    eager = torch.tensor([0.0, 1.0])
    high = torch.tensor([0.001, 1.001])
    candidate = torch.tensor([0.0001, 1.0001])
    assert compare(candidate, eager, high, eager.clone(), gradient=True)["passed"]


def test_probe_loader_rejects_another_checkpoint_before_tensor_use(tmp_path: Path) -> None:
    path = tmp_path / "inputs.pt"
    torch.save(
        {
            "format": "hal.o65.training-probe.v1",
            "provenance": {"checkpoint_sha256": "a" * 64},
            "source_sha256": "b" * 64,
            "training": {},
            "cohorts": [],
        },
        path,
    )
    with pytest.raises(ValueError, match="checkpoint or format mismatch"):
        load_cohort(path, "c" * 64, 0)


def test_probe_loader_rejects_unsupported_clipping(tmp_path: Path) -> None:
    path = tmp_path / "inputs.pt"
    torch.save(
        {
            "format": "hal.o65.training-probe.v1",
            "provenance": {"checkpoint_sha256": "a" * 64},
            "source_sha256": "b" * 64,
            "training": {"clip_epsilon": 0.1},
            "cohorts": [],
        },
        path,
    )
    with pytest.raises(ValueError, match="clip_epsilon"):
        load_cohort(path, "a" * 64, 0)
