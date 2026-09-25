"""Pin the published comparison row used by the KV cache evaluation."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_kv_cache", Path(__file__).resolve().parents[1] / "experiments" / "eval_kv_cache.py"
)
assert _SPEC is not None and _SPEC.loader is not None
eval_kv_cache = importlib.util.module_from_spec(_SPEC)
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
