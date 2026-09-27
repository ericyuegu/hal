"""Pin the published comparison row and named 059 evaluation profiles."""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_kv_cache", Path(__file__).resolve().parents[1] / "scripts" / "eval_kv_cache.py"
)
assert _SPEC is not None and _SPEC.loader is not None
eval_kv_cache = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = eval_kv_cache
_SPEC.loader.exec_module(eval_kv_cache)


def _row() -> dict[str, float | int]:
    return {
        "_step": 1322,
        "eval/checkpoint_step": 131072,
        "eval/boots": 96,
        "eval/crashed": 0,
        "eval/captured_emulator_frames": 691200,
        "eval/prediction_frames": 4,
        "eval/delay_frames": 2,
        "eval/replan_interval_frames": 2,
        "eval/net_stock_per_min": 1.204894549414064,
    }


def _stub_wandb(monkeypatch: pytest.MonkeyPatch, row: dict[str, float | int]) -> None:
    class Run:
        def scan_history(self, *, page_size: int):
            assert page_size == 1500
            return iter((row,))

    class Api:
        def run(self, name: str) -> Run:
            assert name == "ericyuegu/hal/vywk3cih"
            return Run()

    monkeypatch.setattr(eval_kv_cache.wandb, "Api", lambda *, timeout: Api())


def test_baseline_reads_final_p90_history_row(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_wandb(monkeypatch, _row())
    assert eval_kv_cache._baseline()["net_stock_per_min"] == eval_kv_cache.BASELINE_NSM


@pytest.mark.parametrize("changed", ("eval/boots", "eval/delay_frames", "eval/net_stock_per_min"))
def test_baseline_rejects_changed_protocol(monkeypatch: pytest.MonkeyPatch, changed: str) -> None:
    row = _row()
    row[changed] = 0
    _stub_wandb(monkeypatch, row)
    with pytest.raises(ValueError):
        eval_kv_cache._baseline()


@pytest.mark.parametrize(
    ("name", "history", "prefix"),
    (
        ("official-059", "window", 2),
        ("cached-prefix-two", "kv_cache", 2),
        ("local", "kv_cache", 0),
    ),
)
def test_named_profiles_keep_physical_delay_separate_from_fixed_prefix(name: str, history: str, prefix: int) -> None:
    profile = eval_kv_cache.profile_for(name)
    timing = profile.timing
    assert profile.history_mode == history
    assert (timing.physical_delay_frames, timing.inference_allowance_frames) == (0, 0)
    assert (timing.fixed_prefix_frames, timing.replan_interval_frames, timing.prediction_horizon_frames) == (
        prefix,
        2,
        4,
    )
