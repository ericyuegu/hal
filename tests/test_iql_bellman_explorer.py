import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "notebooks" / "iql_bellman_explorer.py"
_SPEC = importlib.util.spec_from_file_location("test_iql_bellman_explorer_module", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
explorer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = explorer
_SPEC.loader.exec_module(explorer)


def test_reward_math_matches_027_stock_damage_and_terminal_semantics() -> None:
    sample = {
        "p1_stock": np.array([4, 4, 3, 3, 0], dtype=np.int32),
        "p2_stock": np.array([4, 3, 3, 0, 0], dtype=np.int32),
        "p1_percent": np.array([0, 10, 10, 30, 0], dtype=np.float32),
        "p2_percent": np.array([0, 0, 25, 25, 0], dtype=np.float32),
    }
    reward = explorer.frame_reward(sample, ego="p1", opp="p2")
    reverse = explorer.frame_reward(sample, ego="p2", opp="p1")

    np.testing.assert_allclose(reward, np.array([0, 0.9, -0.75, 1.3, -1.5], dtype=np.float32))
    np.testing.assert_allclose(reverse, -reward)


def test_returns_and_four_frame_reward_use_t_plus_one_through_t_plus_four() -> None:
    reward = np.arange(8, dtype=np.float32)
    gamma = 0.5

    chunks = explorer.four_frame_rewards(reward, gamma)
    assert chunks[0] == pytest.approx(1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 4)
    assert chunks[2] == pytest.approx(3 + 0.5 * 4 + 0.25 * 5 + 0.125 * 6)

    returns = explorer.discounted_returns(reward, gamma)
    np.testing.assert_allclose(returns[:-1], reward[:-1] + gamma * returns[1:], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("gamma", "expected_percent"),
    [(0.99827, 0.6902), (0.995, 1.9850), (0.99, 3.9404)],
)
def test_four_frame_contraction_reference_values(gamma: float, expected_percent: float) -> None:
    assert 100 * explorer.four_frame_contraction(gamma) == pytest.approx(expected_percent, abs=5e-4)


def test_rendered_page_is_self_contained_and_exposes_the_planned_controls() -> None:
    game = {
        "file": "example.slp",
        "source": "human (ranked)",
        "tier": "diamond",
        "p1": "FOX",
        "p2": "FALCO",
        "stage": "BATTLEFIELD",
        "frame0": -123,
        "note": "final 2–0",
        "p1Percent": [0.0] * 8,
        "p2Percent": [0.0] * 8,
        "p1Stock": [4] * 8,
        "p2Stock": [4] * 8,
    }
    html = explorer.render_html([game], slippilab_base="http://localhost:5173", slippilab_mount="hal-iql")

    assert "__DATA__" not in html
    assert "__SLIPPILAB_BASE__" not in html
    assert '"file":"example.slp"' in html
    assert 'id="gamma"' in html
    assert 'id="tau"' in html
    assert 'id="sigma"' in html
    assert 'id="support"' in html
    assert 'id="alpha"' in html
    assert 'id="updates"' in html
    assert "Bellman contraction" in html
    assert "expectile / noise push" in html
    assert "support clipping" in html


def test_render_requires_at_least_one_replay() -> None:
    with pytest.raises(ValueError, match="at least one replay"):
        explorer.render_html([], slippilab_base="http://localhost:5173", slippilab_mount="hal-iql")
