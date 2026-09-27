"""Gameplay protocol choices must survive the standalone evaluation entrypoint."""

import importlib.util
import sys
from pathlib import Path
from typing import cast

import pytest

from hal.inference.action_sequence_policy import ActionSequencePolicy
from hal.inference.api import RuntimeConfig

_SPEC = importlib.util.spec_from_file_location(
    "hal_eval_kv_cache", Path(__file__).parents[1] / "scripts" / "eval_kv_cache.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _Policy:
    context_frames = 256

    def __init__(self) -> None:
        self.seed: int | None = None

    def reset_prediction(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed


@pytest.mark.parametrize(
    ("name", "observed_actions", "prefix"),
    [("official-059", False, 2), ("cached-prefix-two", True, 2), ("local", True, 0)],
)
def test_evaluation_factory_preserves_action_history_and_checkpoint_return(
    name: str, observed_actions: bool, prefix: int
) -> None:
    profile = _MODULE.profile_for(name)
    policy = _Policy()
    runtime = RuntimeConfig(1, (0,), replan_interval_frames=2)
    p90 = 19.9760597229004
    factory = _MODULE.EvaluationPolicyFactory(
        cast(ActionSequencePolicy, policy), runtime, profile.timing, p90, profile.observed_actions
    )

    first = factory()

    assert first.runtime_spec.observed_actions is observed_actions
    assert first.runtime_spec.committed_frames == prefix
    assert first.runtime_spec.execution_stride == 2
    assert first.runtime_spec.prediction_frames == 4
    assert first.timing.physical_delay_frames == 0
    assert first.default_settings.desired_return == p90
    assert policy.seed == 0

    second = factory()
    assert policy.seed == 1
    assert second.default_settings == first.default_settings
