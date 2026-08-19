"""Compute paired boot-cluster effects for experiment 037."""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import tyro

from hal.eval.cross_stage import BOOTSTRAP_RESAMPLES
from hal.eval.cross_stage import FRAMES_PER_MINUTE
from hal.eval.cross_stage import MatchRow

_CELLS = ("w0d0", "w0d1", "w1d0", "w1d1")
_METRICS = ("net_stock_per_min", "net_dmg_per_min")


@dataclass(frozen=True, slots=True)
class Effect:
    """One paired factorial contrast and percentile interval."""

    mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True, slots=True)
class Args:
    w0d0: Path
    w0d1: Path
    w1d0: Path
    w1d1: Path
    output: Path | None = None
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES
    seed: int = 0


def _rows_by_boot(rows: Sequence[MatchRow]) -> dict[int, list[MatchRow]]:
    by_boot: dict[int, list[MatchRow]] = {}
    for row in rows:
        by_boot.setdefault(row.boot_index, []).append(row)
    if not by_boot:
        raise ValueError("each cell must contain at least one completed boot")
    for boot, values in by_boot.items():
        matchups = {(row.ego_character, row.opp_character) for row in values}
        if len(matchups) != 1:
            raise ValueError(f"boot {boot} contains multiple character matchups: {sorted(matchups)}")
        if not any(row.active_frames > 0 for row in values):
            raise ValueError(f"boot {boot} contains no active frame")
    return by_boot


def _validate_schedules(by_cell: Mapping[str, dict[int, list[MatchRow]]]) -> list[int]:
    if set(by_cell) != set(_CELLS):
        raise ValueError(f"expected exactly cells {_CELLS}, got {sorted(by_cell)}")
    reference = by_cell["w0d0"]
    boots = sorted(reference)
    for cell in _CELLS[1:]:
        current = by_cell[cell]
        if sorted(current) != boots:
            raise ValueError(f"{cell} boot IDs do not match w0d0")
        for boot in boots:
            reference_matchup = (reference[boot][0].ego_character, reference[boot][0].opp_character)
            current_matchup = (current[boot][0].ego_character, current[boot][0].opp_character)
            if current_matchup != reference_matchup:
                raise ValueError(
                    f"{cell} boot {boot} matchup {current_matchup} does not match {reference_matchup}"
                )
    return boots


def _boot_components(rows: Sequence[MatchRow]) -> tuple[float, float, float]:
    active = [row for row in rows if row.active_frames > 0]
    minutes = sum(row.active_frames for row in active) / FRAMES_PER_MINUTE
    stock = sum(row.stocks_taken - row.stocks_lost for row in active)
    damage = sum(row.damage_dealt - row.damage_taken for row in active)
    return float(stock), float(damage), minutes


def _rates(components: np.ndarray, indices: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    selected = components if indices is None else components[indices]
    axis = 0 if indices is None else 1
    denominators = selected[..., 2].sum(axis=axis)
    if np.any(denominators <= 0):
        raise ValueError("a paired bootstrap sample has no active time")
    return {
        "net_stock_per_min": selected[..., 0].sum(axis=axis) / denominators,
        "net_dmg_per_min": selected[..., 1].sum(axis=axis) / denominators,
    }


def _contrasts(cell_values: Mapping[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    y00 = cell_values["w0d0"]
    y01 = cell_values["w0d1"]
    y10 = cell_values["w1d0"]
    y11 = cell_values["w1d1"]
    return {
        "width": ((y10 - y00) + (y11 - y01)) / 2,
        "decoder": ((y01 - y00) + (y11 - y10)) / 2,
        "interaction": (y11 - y10) - (y01 - y00),
    }


def analyze_factorial(
    rows_by_cell: Mapping[str, Sequence[MatchRow]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, object]:
    """Return cell rates and paired W, D, and interaction effects."""
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    grouped = {cell: _rows_by_boot(rows) for cell, rows in rows_by_cell.items()}
    boots = _validate_schedules(grouped)
    components = {
        cell: np.asarray([_boot_components(grouped[cell][boot]) for boot in boots], dtype=np.float64)
        for cell in _CELLS
    }
    indices = np.random.default_rng(seed).integers(0, len(boots), size=(bootstrap_resamples, len(boots)))
    point_rates = {cell: _rates(values) for cell, values in components.items()}
    bootstrap_rates = {cell: _rates(values, indices) for cell, values in components.items()}
    output: dict[str, object] = {
        "schema_version": 1,
        "boots": len(boots),
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "cells": {
            cell: {metric: float(cast(float, point_rates[cell][metric])) for metric in _METRICS}
            for cell in _CELLS
        },
        "effects": {},
    }
    effects = cast(dict[str, object], output["effects"])
    for metric in _METRICS:
        points = _contrasts({cell: cast(float, point_rates[cell][metric]) for cell in _CELLS})
        samples = _contrasts({cell: cast(np.ndarray, bootstrap_rates[cell][metric]) for cell in _CELLS})
        metric_effects: dict[str, dict[str, float]] = {}
        for name in ("width", "decoder", "interaction"):
            low, high = np.percentile(cast(np.ndarray, samples[name]), (2.5, 97.5))
            metric_effects[name] = asdict(
                Effect(mean=float(points[name]), ci_low=float(low), ci_high=float(high))
            )
        effects[metric] = metric_effects
    return output


def _load_rows(path: Path) -> tuple[list[MatchRow], dict[str, object]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 6:
        raise ValueError(f"{path}: expected match-row schema 6")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: missing protocol")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing rows")
    return [MatchRow.from_dict(row) for row in rows], protocol


def _validate_protocols(protocols: Mapping[str, Mapping[str, object]]) -> None:
    fields = (
        "n_matchups",
        "max_frames",
        "seed",
        "cpu_level",
        "ego_port",
        "seed_stage",
        "matchup_schedule_sha256",
        "exec_horizon",
        "bootstrap_resamples",
    )
    reference = protocols["w0d0"]
    for cell in _CELLS[1:]:
        changed = {name: (protocols[cell].get(name), reference.get(name)) for name in fields if protocols[cell].get(name) != reference.get(name)}
        if changed:
            raise ValueError(f"{cell} evaluation protocol does not match w0d0: {changed}")


def main(args: Args) -> None:
    paths = {cell: getattr(args, cell) for cell in _CELLS}
    loaded = {cell: _load_rows(path) for cell, path in paths.items()}
    _validate_protocols({cell: protocol for cell, (_, protocol) in loaded.items()})
    output = analyze_factorial(
        {cell: rows for cell, (rows, _) in loaded.items()},
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
