"""Units and uncertainty checks for the executed-tail artifact analysis."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "tail_findings.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_tail_findings", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
findings = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = findings
_SPEC.loader.exec_module(findings)


def test_constant_paired_effect_has_degenerate_replay_interval() -> None:
    result = findings.paired_interval(np.full(256, 0.25))
    assert result["mean_bits_per_executed_frame"] == pytest.approx(0.25)
    assert result["replay_bootstrap_95"] == pytest.approx([0.25, 0.25])
    assert result["includes_monte_carlo_error"] is False


@pytest.mark.parametrize("values", ([0.0], [float("nan")] * 256, [True] * 256))
def test_invalid_replay_score_vector_is_rejected(values: list[object]) -> None:
    with pytest.raises(ValueError):
        findings.replay_vector(values)


def test_required_six_checkpoint_set_is_not_optional(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="all six"):
        findings.summarize(tmp_path)
