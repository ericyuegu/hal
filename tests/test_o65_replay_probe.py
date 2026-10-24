"""Keep precision errors diagnostic while rejecting nonfinite results."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from test_065_loop import exp
from test_065_loop import transformer_trainer

from hal.eval.self_play_checkpoint import EvaluationCheckpoint
from hal.models.attention import Trunk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts import o65_replay_probe as probe
    from scripts.o65_replay_probe import compare
    from scripts.o65_replay_probe import load_cohort
finally:
    sys.path.pop(0)


def test_gradient_direction_change_is_reported_without_a_precision_gate() -> None:
    eager = torch.tensor([1.0, 0.0])
    high = torch.tensor([0.0, 1.0])
    candidate = torch.tensor([0.7, 0.7])
    report = compare(candidate, eager, high, eager.clone())
    assert report["compiled"]["relative_l2"] < 1.5 * report["eager"]["relative_l2"]
    assert report["cosine"] < 0.8
    assert report["passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_precision_diagnostic_rejects_nonfinite_outputs(value: float) -> None:
    eager = torch.ones(2)
    assert not compare(torch.tensor([value, 1]), eager, eager, eager)["passed"]


def test_gradient_gate_accepts_small_error_with_the_same_direction() -> None:
    eager = torch.tensor([0.0, 1.0])
    high = torch.tensor([0.001, 1.001])
    candidate = torch.tensor([0.0001, 1.0001])
    assert compare(candidate, eager, high, eager.clone())["passed"]


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_eager_precision_control_runs_before_any_compiled_decoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run = transformer_trainer("cpu", decoder_precision="bf16")
    prepared = run.prepare(run.engine.collect(0))
    windows = prepared.windows
    index = int(torch.nonzero(windows.slots.valid_WK)[0, 0])
    position = int(torch.nonzero(windows.slots.valid_WK[index])[0, 0])
    part = windows.take(torch.tensor([index]), torch.full((len(windows.rows), 1), position), torch.tensor([True]))
    context, observed = run.policy._packed_context(part.inputs()[:5])
    checkpoint = EvaluationCheckpoint(
        copy.deepcopy(run.policy.model).cuda(), run.policy.statistics, "a" * 64, "b" * 64, {}
    )
    monkeypatch.setattr(probe, "load_evaluation_checkpoint", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        probe,
        "load_cohort",
        lambda *_args: {
            "context": context,
            "observed": observed,
            "inputs": part.inputs(),
            "selected": part.slots.valid_WK,
            "coefficients": torch.zeros_like(part.rows, dtype=torch.float32),
        },
    )
    decoder = probe.replay_windows

    class CompiledDecoderReached(Exception):
        pass

    def compile_operation(operation: object, **_kwargs: object) -> object:
        if operation is decoder:

            def compiled(*_args: object, **_kwargs: object) -> None:
                raise CompiledDecoderReached

            return compiled

        if operation is not probe.history:
            return operation

        def compiled_history(trunk: Trunk, tokens: torch.Tensor) -> torch.Tensor:
            return probe.history(trunk, tokens)

        return compiled_history

    monkeypatch.setattr(torch, "compile", compile_operation)
    with pytest.raises(CompiledDecoderReached):
        probe.main(probe.Args(tmp_path / "inputs.pt"))
    printed = capsys.readouterr().out
    assert "eager reference" in printed
    assert "compiled reference" in printed
