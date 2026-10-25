"""Keep precision errors diagnostic while rejecting nonfinite results."""

import sys
from pathlib import Path

import pytest
import torch
from test_065_loop import exp
from test_065_loop import transformer_trainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts import o65_replay_probe as probe
    from scripts.o65_replay_probe import compare
    from scripts.o65_replay_probe import load_cohort
finally:
    sys.path.pop(0)


def test_finite_gradient_report_exposes_direction_change() -> None:
    eager = torch.tensor([1.0, 0.0])
    high = torch.tensor([0.0, 1.0])
    candidate = torch.tensor([0.7, 0.7])
    report = compare(candidate, eager, high, eager.clone())
    assert report["cosine"] == pytest.approx(2**-0.5)
    assert report["eager_fp32_cosine"] == 0
    assert report["compiled_fp32_cosine"] == pytest.approx(2**-0.5)
    assert report["finite"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_precision_diagnostic_rejects_nonfinite_outputs(value: float) -> None:
    eager = torch.ones(2)
    assert not compare(torch.tensor([value, 1]), eager, eager, eager)["finite"]


def test_history_matches_the_training_boundary_and_input_gradient() -> None:
    run = transformer_trainer("cpu", decoder_precision="fp32")
    tokens = torch.randn(2, 256, run.policy.model.cfg.d_model, requires_grad=True)
    expected = exp.replay_history(run.policy.model.trunk, tokens)
    actual = probe.history(run.policy.model.trunk, tokens)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    expected_gradient = torch.autograd.grad(expected, tokens, upstream)[0]
    actual_gradient = torch.autograd.grad(actual, tokens, upstream)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


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
