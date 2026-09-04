"""Parity contract for the direct experiment-051 rewrite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "o51_v5_characterization.json"
CHARACTERIZE = ROOT / "tests" / "fixtures" / "o51_v5_characterize.py"
RESULT_PREFIX = "O51_V5_CHARACTERIZATION="


def _current_contract() -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(CHARACTERIZE)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payloads = [
        line.removeprefix(RESULT_PREFIX) for line in result.stdout.splitlines() if line.startswith(RESULT_PREFIX)
    ]
    assert len(payloads) == 1, result.stdout
    value = json.loads(payloads[0])
    assert isinstance(value, dict)
    return value


def test_o51_v5_contract_matches_the_isolated_baseline() -> None:
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["schema"] == 1
    assert len(fixture["provenance"]["git_commit"]) == 40
    assert all(len(digest) == 64 for digest in fixture["provenance"]["sources"].values())

    expected = fixture["contract"]
    actual = _current_contract()
    expected_numeric = expected.pop("numeric")
    actual_numeric = actual.pop("numeric")

    assert actual == expected
    assert actual_numeric["hidden_shape"] == expected_numeric["hidden_shape"]
    assert actual_numeric["metric_keys"] == expected_numeric["metric_keys"]
    # CPU kernels can change reduction order without changing the experiment.
    for name in (
        "hidden_l2",
        "hidden_mean",
        "loss",
        "nll_mean",
        "policy_loss_bits",
        "value_loss",
    ):
        assert actual_numeric[name] == pytest.approx(expected_numeric[name], rel=2e-5, abs=2e-6)
