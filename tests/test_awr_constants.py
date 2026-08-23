"""Tests for Experiment 040's deterministic AWR calibration sample."""

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "040_awr_constants.py"
_SPEC = importlib.util.spec_from_file_location("test_awr_constants_notebook", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
constants = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = constants
_SPEC.loader.exec_module(constants)


def test_stratified_spans_are_exact_deterministic_and_dispersed() -> None:
    first = constants.stratified_spans(1_000, 123, np.random.default_rng(7), max_spans=8)
    second = constants.stratified_spans(1_000, 123, np.random.default_rng(7), max_spans=8)

    assert first == second
    assert len(first) == 8
    assert sum(stop - start for start, stop in first) == 123
    assert all(0 <= start < stop <= 1_000 for start, stop in first)
    assert all(left_stop <= right_start for (_, left_stop), (right_start, _) in itertools.pairwise(first))
    assert first[0][0] < 125
    assert first[-1][0] >= 875


def test_stratified_spans_handle_empty_and_full_samples() -> None:
    rng = np.random.default_rng(0)

    assert constants.stratified_spans(10, 0, rng) == ()
    spans = constants.stratified_spans(10, 10, rng, max_spans=8)
    assert sum(stop - start for start, stop in spans) == 10
    assert spans[0][0] == 0
    assert spans[-1][1] == 10
