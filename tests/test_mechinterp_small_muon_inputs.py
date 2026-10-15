from dataclasses import replace

import numpy as np
import pytest

from research.mechinterp_20260911 import small_muon_inputs


def _source() -> dict[str, object]:
    return {
        "schema_version": np.asarray(7, dtype=np.int32),
        "frame": np.arange(300, dtype=np.int32),
        "p1_position_x": np.arange(300, dtype=np.float32),
        "p2_position_x": -np.arange(300, dtype=np.float32),
    }


def _identity(source: dict[str, object]) -> small_muon_inputs.WindowIdentity:
    return small_muon_inputs.WindowIdentity(
        shard=0,
        row=0,
        start=10,
        length=260,
        ego="p1",
        raw_sha256=small_muon_inputs.hash_source_window(source, 10, 270),
        first_frame=10,
    )


def test_restored_context_keeps_same_query_and_final_128_frames() -> None:
    source = _source()
    window = small_muon_inputs.context_window(source, _identity(source))
    np.testing.assert_array_equal(window["ego_position_x"], np.arange(138, 266, dtype=np.float32))
    assert np.asarray(window["ego_position_x"])[-1] == 10 + 255
    assert int(window["ctx_pad"]) == 0


def test_restored_context_preserves_the_recorded_ego_port() -> None:
    source = _source()
    window = small_muon_inputs.context_window(source, replace(_identity(source), ego="p2"))
    np.testing.assert_array_equal(window["ego_position_x"], -np.arange(138, 266, dtype=np.float32))


def test_restored_context_checks_original_window_before_cropping() -> None:
    source = _source()
    identity = _identity(source)
    source["p1_position_x"] = np.asarray(source["p1_position_x"]).copy()
    source["p1_position_x"][20] = -123.0
    with pytest.raises(ValueError, match="260-frame raw window hash"):
        small_muon_inputs.context_window(source, identity)


def test_restored_context_rejects_a_restart_even_with_matching_hash() -> None:
    source = _source()
    source["frame"] = np.asarray(source["frame"]).copy()
    source["frame"][200] = 0
    with pytest.raises(ValueError, match="frame gap or restart"):
        small_muon_inputs.context_window(source, _identity(source))


def test_source_hash_retains_scalar_schema() -> None:
    source = _source()
    before = small_muon_inputs.hash_source_window(source, 10, 270)
    source["schema_version"] = np.asarray(6, dtype=np.int32)
    assert small_muon_inputs.hash_source_window(source, 10, 270) != before
