"""Independent enumeration and limits for the IID-data counterexample."""

import importlib.util
import itertools
import math
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "research/mechinterp_20260911/iid_data_counterexample.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_iid_data", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
iid_data = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = iid_data
_SPEC.loader.exec_module(iid_data)


def test_binomial_sum_matches_independent_ordered_dataset_enumeration() -> None:
    expected_nll = 0.0
    expected_success = 0.0
    for labels in itertools.product((0, 1), repeat=4):
        probability = math.prod(0.9 if label else 0.1 for label in labels)
        fitted = (sum(labels) + 0.01) / 4.02
        expected_nll += probability * (-0.9 * math.log(fitted) - 0.1 * math.log1p(-fitted))
        expected_success += probability * fitted**20
    result = iid_data.expected_metrics(4)
    assert result.nll_nats_per_action == pytest.approx(expected_nll)
    assert result.task_success == pytest.approx(expected_success)
    assert result.training_count_probability_mass == pytest.approx(1)


def test_more_iid_data_lowers_expected_nll_and_expected_success() -> None:
    rows = [iid_data.expected_metrics(count) for count in iid_data.SAMPLE_COUNTS]
    for earlier, later in itertools.pairwise(rows):
        assert later.nll_nats_per_action < earlier.nll_nats_per_action
        assert later.task_success < earlier.task_success
        assert later.action_correct_probability > earlier.action_correct_probability
    assert rows[0].task_success > 0.7
    assert rows[-1].task_success < 0.14
    assert rows[-1].nll_nats_per_action > iid_data.population_nll(0.9)


@pytest.mark.parametrize("examples", (0, -1, 513))
def test_rejects_unsupported_dataset_sizes(examples: int) -> None:
    with pytest.raises(ValueError):
        iid_data.expected_metrics(examples)
