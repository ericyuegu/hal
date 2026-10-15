import pytest

from research.mechinterp_20260911 import small_muon_inventory


def _rows() -> list[dict[str, object]]:
    return [
        {"global_step": step, "train/loss": 2.0, "train/grad_norm": gradient}
        for step, gradient in ((512, 0.6), (1024, 0.4), (12288, 0.3), (14336, 0.2), (16383, 0.1))
    ]


def test_history_summary_uses_global_steps_and_exact_windows() -> None:
    rows = _rows()
    result = small_muon_inventory.summarize_history(list(reversed(rows)))
    assert result["first_global_step"] == 512
    assert result["last_global_step"] == 16383
    assert result["all_finite"] is True
    assert result["updates_14336_16383"] == {
        "rows": 2,
        "loss": {"min": 2.0, "median": 2.0, "max": 2.0},
        "gradient_norm": {"min": 0.1, "median": 0.15000000000000002, "max": 0.2},
    }


def test_history_summary_rejects_missing_window() -> None:
    with pytest.raises(ValueError, match="No training rows"):
        small_muon_inventory.summarize_history(_rows()[:2])


def test_history_summary_does_not_call_nonfinite_history_healthy() -> None:
    rows = _rows()
    rows[-1]["train/grad_norm"] = float("inf")
    assert small_muon_inventory.summarize_history(rows) == {"rows": 5, "all_finite": False}


def test_history_summary_rejects_wandb_row_index_as_training_step() -> None:
    rows = _rows()
    rows[0]["_step"] = rows[0].pop("global_step")
    with pytest.raises(ValueError, match="global_step"):
        small_muon_inventory.summarize_history(rows)
