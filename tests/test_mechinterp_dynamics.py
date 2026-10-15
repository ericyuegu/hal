"""Regression tests for exact-step analysis of sparse W&B histories."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "research/mechinterp_20260911/dynamics.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_dynamics", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
dynamics = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dynamics
_SPEC.loader.exec_module(dynamics)


def test_history_collapses_only_exact_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    row = {"global_step": 8192, "val/nll": 1.5}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    assert dynamics.history(path) == {8192: row}
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "val/nll": 1.4}) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        dynamics.history(path)


def test_alignment_requires_exact_step() -> None:
    with pytest.raises(ValueError, match="interpolation"):
        dynamics.align_evaluations({8192: {}}, {8191: {}})


def test_clipping_uses_window_fraction_not_mean_gradient_threshold() -> None:
    training = {
        25: {"optimizer/grad_norm": 4.0, "optimizer/clip_fraction": 1.0},
        28925: {"optimizer/grad_norm": 0.35, "optimizer/clip_fraction": 0.0},
        28950: {"optimizer/grad_norm": 0.369, "optimizer/clip_fraction": 0.04},
    }
    assert dynamics.clipped_windows(training, warmup=4096) == [training[28950]]


def test_alignment_checks_logged_objective_identity() -> None:
    evaluation = {
        "eval/checkpoint_step": 8192,
        "eval/boots": 96,
        "eval/crashed": 0,
        "eval/net_stock_per_min": 0.3,
        "eval/net_dmg_per_min": 30.0,
        "eval/net_stock_lcb": 0.2,
    }
    validation = {
        "val/nll": 4 / 3,
        "val/near_nll": 1.0,
        "val/far_nll": 2.0,
        "val/change_f1": 0.25,
        "val/exact_frame_acc": 0.6,
        "val/sequence_acc": 0.4,
    }
    rows = dynamics.align_evaluations({8192: evaluation}, {8192: validation})
    assert rows[0].stock_rate == 0.3
    validation["val/nll"] = 1.5
    with pytest.raises(ValueError, match="do not reproduce"):
        dynamics.align_evaluations({8192: evaluation}, {8192: validation})
