"""Use the timed architecture FLOPs when reporting training compute floors."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts.o65_training_trace import phase_floors
finally:
    sys.path.pop(0)


def write_profile(path: Path) -> None:
    (path / "resolved.json").write_text(
        json.dumps(
            {"format": "hal.o65.gpu-profile-inputs.v1", "gpu": "NVIDIA B200", "environment": {"world_size": "1"}}
        )
    )
    metrics = {}
    for name in ("preparation", "actor", "critic"):
        metrics[f"systems/{name}_seconds"] = 4.0
        metrics[f"flops/{name}_total"] = 2.25e15
    (path / "timed.json").write_text(
        json.dumps({"format": "hal.o65.gpu-profile.v1", "mode": "timed", "metrics": metrics})
    )


def test_floors_use_architecture_flops_and_untraced_phase_times(tmp_path: Path) -> None:
    write_profile(tmp_path)
    rows = phase_floors(tmp_path)
    assert tuple(row.name for row in rows) == ("preparation", "actor", "critic")
    for row in rows:
        assert row.floor_seconds == 1.0
        assert row.mfu == 0.25


def test_floors_reject_missing_architecture_flops(tmp_path: Path) -> None:
    write_profile(tmp_path)
    timed = json.loads((tmp_path / "timed.json").read_text())
    del timed["metrics"]["flops/actor_total"]
    (tmp_path / "timed.json").write_text(json.dumps(timed))
    with pytest.raises(KeyError, match="flops/actor_total"):
        phase_floors(tmp_path)


def test_floors_reject_distributed_work_with_a_single_gpu_peak(tmp_path: Path) -> None:
    write_profile(tmp_path)
    resolved = json.loads((tmp_path / "resolved.json").read_text())
    resolved["environment"]["world_size"] = "8"
    (tmp_path / "resolved.json").write_text(json.dumps(resolved))
    with pytest.raises(ValueError, match="single-GPU profile"):
        phase_floors(tmp_path)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_floors_reject_invalid_times(tmp_path: Path, seconds: float) -> None:
    write_profile(tmp_path)
    timed = json.loads((tmp_path / "timed.json").read_text())
    timed["metrics"]["systems/actor_seconds"] = seconds
    (tmp_path / "timed.json").write_text(json.dumps(timed))
    with pytest.raises(ValueError, match="invalid timed work for actor"):
        phase_floors(tmp_path)
