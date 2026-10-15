"""Focused tests for the fixed, audit-only learner replay path."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "research/mechinterp_20260911/live_replay_audit.py"
SPEC = importlib.util.spec_from_file_location("live_replay_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_context_selection_is_explicit() -> None:
    assert MODULE.CONDITIONS == ("early_24576", "slump_57344")
    assert MODULE.BOOT_INDICES == (0, 1, 2)
    assert MODULE.WINDOW_START == 700
    assert MODULE.CONTEXT_LENGTH == 256
    assert MODULE.CONTROLLED_PORT == 1
    assert MODULE.MASKED_PLAYER_ID == 0


def test_validate_frame_ids_rejects_restart_boundary() -> None:
    frame = np.arange(MODULE.WINDOW_START + MODULE.CONTEXT_LENGTH, dtype=np.int32)
    frame[-1] = frame[-2] - 1
    with pytest.raises(ValueError, match="restart"):
        MODULE.validate_frame_ids(frame)


def test_validate_frame_ids_rejects_short_replay() -> None:
    frame = np.arange(MODULE.WINDOW_START + MODULE.CONTEXT_LENGTH - 1, dtype=np.int32)
    with pytest.raises(ValueError, match="need at least"):
        MODULE.validate_frame_ids(frame)


def test_replay_path_rejects_unplanned_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported condition"):
        MODULE.replay_path(tmp_path, "treatment", 0)
    with pytest.raises(ValueError, match="unsupported boot"):
        MODULE.replay_path(tmp_path, "early_24576", 3)


def test_controlled_port_check_requires_schedule_match(monkeypatch: pytest.MonkeyPatch) -> None:
    length = MODULE.WINDOW_START + MODULE.CONTEXT_LENGTH
    source = {
        "p1_character": np.full(length, 1, dtype=np.int32),
        "p2_character": np.full(length, 22, dtype=np.int32),
    }
    monkeypatch.setattr(MODULE, "extract_replay", lambda _path: source)
    monkeypatch.setattr(
        MODULE,
        "start_port_record",
        lambda _path: [
            {"libmelee_port": 1, "peppi_port": 0, "character_external": 2, "player_type": 0},
            {"libmelee_port": 2, "peppi_port": 1, "character_external": 20, "player_type": 1},
        ],
    )
    record = MODULE.verify_controlled_port(Path("ignored.slp"), expected_ego="FOX", expected_opponent="FALCO")
    assert record["controlled_port"] == 1
    assert record["p1_maps_to_controlled_port"] is True
    with pytest.raises(ValueError, match="differs"):
        MODULE.verify_controlled_port(Path("ignored.slp"), expected_ego="MARTH", expected_opponent="FALCO")
