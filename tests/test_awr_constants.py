"""Tests for Experiment 040's deterministic AWR calibration sample."""

import importlib.util
import itertools
import json
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


def test_checked_in_calibration_artifact_is_complete() -> None:
    artifact_path = _PATH.with_suffix(".json")
    artifact = json.loads(artifact_path.read_text())

    assert artifact["sample_replays"] == constants.SAMPLE_REPLAYS
    assert artifact["sample_seed"] == constants.SEED
    assert artifact["loader_geometry"] == {
        "L_chunk": constants.L_CHUNK,
        "L_ctx": constants.L_CTX,
        "windows_per_replay": constants.WINDOWS_PER_REPLAY,
    }
    assert artifact["reward"] == {
        "damage_shaping": constants.DAMAGE_SHAPING,
        "gamma": constants.GAMMA,
        "stock_value": constants.STOCK_VALUE,
        "win_reward": constants.WIN_REWARD,
    }
    assert artifact["awr_beta"] == constants.BETA
    assert artifact["awr_weight_max"] == constants.WEIGHT_MAX
    assert sum(source["requested_replays"] for source in artifact["sources"]) == constants.SAMPLE_REPLAYS
    assert sum(source["terminal_replays"] for source in artifact["sources"]) == artifact["terminal_replays"]
    assert sum(source["eligible_positions"] for source in artifact["sources"]) == artifact["eligible_positions"]
    assert [source["name"] for source in artifact["sources"]] == [
        source.name for source in constants.streams.POLICY_WORLD_V7_SOURCES
    ]


def test_checked_in_calibration_passes_acceptance_gates() -> None:
    artifact = json.loads(_PATH.with_suffix(".json").read_text())

    assert artifact["weight_clip_fraction"] == 0.0
    assert artifact["weight_ess"] > 0.95
    assert 0.78 < artifact["between_256_frame_window_variance_fraction"] < 0.80
