import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "metric_counterexamples.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_counterexamples", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
metric_counterexamples = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = metric_counterexamples
_SPEC.loader.exec_module(metric_counterexamples)


@pytest.mark.parametrize("probability", (0.15, 0.5, 0.8, 0.99))
@pytest.mark.parametrize("horizon", (2, 4, 6))
def test_correlated_metric_formulas_match_full_enumeration(probability: float, horizon: int) -> None:
    formula = metric_counterexamples.correlated_actions(probability, horizon)
    enumerated = metric_counterexamples.enumerate_correlated_actions(probability, horizon)
    for name, value in enumerated.items():
        assert formula[name] == pytest.approx(value)


def test_independent_policy_has_no_mixed_prefix_gap() -> None:
    result = metric_counterexamples.correlated_actions(0.5, 6)
    assert result["gap"] == pytest.approx(0)
    assert result["success_probability"] == pytest.approx(2 / 64)


def test_fully_supported_state_action_pairs_can_reverse_nll_and_success_order() -> None:
    first = metric_counterexamples.unequal_consequences(0.7, 0.99, 0.05)
    second = metric_counterexamples.unequal_consequences(0.99, 0.1, 0.05)
    assert second["nll"] < first["nll"]
    assert second["success"] < first["success"]


def test_equal_consequence_loss_is_exact_population_cross_entropy() -> None:
    result = metric_counterexamples.unequal_consequences(0.5, 0.5, 0.3, expert_preference=0.5)
    assert result["nll"] == pytest.approx(0.6931471805599453)
    assert result["success"] == pytest.approx(0.85)


def test_beating_a_fallible_expert_can_increase_population_nll() -> None:
    expert_match = metric_counterexamples.imperfect_demonstrator(0.99)
    better_policy = metric_counterexamples.imperfect_demonstrator(0.9999)
    assert better_policy["nll"] > expert_match["nll"]
    assert better_policy["success"] > expert_match["success"]
    assert expert_match["nll"] == pytest.approx(0.05600153435484737)
    assert better_policy["nll"] == pytest.approx(0.09220240867009361)
