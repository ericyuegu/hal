"""Tests for the paired experiment-037 factorial reduction."""

import importlib.util
import sys
from pathlib import Path

import pytest

from hal.eval.cross_stage import MatchRow

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_037_factorial.py"
_SPEC = importlib.util.spec_from_file_location("test_analyze_037", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def _row(
    boot: int,
    *,
    stock: int,
    damage: float,
    ego_character: int = 1,
    opp_character: int = 2,
) -> MatchRow:
    return MatchRow(
        ego_character=ego_character,
        opp_character=opp_character,
        stage=31,
        boot_index=boot,
        match_ordinal=0,
        active_frames=3_600,
        total_frames=3_723,
        damage_dealt=max(damage, 0.0),
        damage_taken=max(-damage, 0.0),
        stocks_taken=max(stock, 0),
        stocks_lost=max(-stock, 0),
    )


def _rows(stock: int, damage: float) -> list[MatchRow]:
    return [_row(boot, stock=stock, damage=damage) for boot in range(8)]


def test_known_main_effects_and_interaction_are_recovered() -> None:
    rows = {
        "w0d0": _rows(0, 0.0),
        "w0d1": _rows(1, 10.0),
        "w1d0": _rows(2, 20.0),
        "w1d1": _rows(4, 40.0),
    }
    output = analysis.analyze_factorial(rows, bootstrap_resamples=100, seed=7)
    stock = output["effects"]["net_stock_per_min"]
    damage = output["effects"]["net_dmg_per_min"]
    assert stock["width"] == {"mean": 2.5, "ci_low": 2.5, "ci_high": 2.5}
    assert stock["decoder"] == {"mean": 1.5, "ci_low": 1.5, "ci_high": 1.5}
    assert stock["interaction"] == {"mean": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    assert damage["width"]["mean"] == 25.0
    assert damage["decoder"]["mean"] == 15.0
    assert damage["interaction"]["mean"] == 10.0


def test_bootstrap_is_deterministic() -> None:
    rows = {
        cell: [_row(boot, stock=(boot + index) % 3 - 1, damage=float(boot * index)) for boot in range(12)]
        for index, cell in enumerate(analysis._CELLS)
    }
    first = analysis.analyze_factorial(rows, bootstrap_resamples=200, seed=11)
    second = analysis.analyze_factorial(rows, bootstrap_resamples=200, seed=11)
    assert first == second


def test_missing_cell_or_boot_is_rejected() -> None:
    rows = {cell: _rows(0, 0.0) for cell in analysis._CELLS}
    del rows["w1d1"]
    with pytest.raises(ValueError, match="expected exactly cells"):
        analysis.analyze_factorial(rows)

    rows = {cell: _rows(0, 0.0) for cell in analysis._CELLS}
    rows["w1d1"].pop()
    with pytest.raises(ValueError, match="boot IDs"):
        analysis.analyze_factorial(rows)


def test_mismatched_character_schedule_is_rejected() -> None:
    rows = {cell: _rows(0, 0.0) for cell in analysis._CELLS}
    rows["w0d1"][3] = _row(3, stock=0, damage=0.0, ego_character=7, opp_character=8)
    with pytest.raises(ValueError, match="matchup"):
        analysis.analyze_factorial(rows)


def test_protocol_mismatch_is_rejected() -> None:
    protocol = {
        "n_matchups": 96,
        "max_frames": 7_200,
        "seed": 0,
        "cpu_level": 9,
        "ego_port": 1,
        "seed_stage": 31,
        "matchup_schedule_sha256": "a" * 64,
        "exec_horizon": 4,
        "bootstrap_resamples": 2_000,
    }
    protocols = {cell: dict(protocol) for cell in analysis._CELLS}
    protocols["w1d0"]["seed"] = 1
    with pytest.raises(ValueError, match="protocol"):
        analysis._validate_protocols(protocols)
