"""Tests for the O52 replay value-meter data path."""

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from hal.scripts import value_meter


def _sample(length: int = 5) -> dict[str, np.ndarray]:
    return {
        "frame": np.arange(-123, -123 + length, dtype=np.int32),
        "p1_position_x": np.arange(length, dtype=np.float32) + 10,
        "p2_position_x": np.arange(length, dtype=np.float32) + 20,
        "p1_button_a": np.arange(length, dtype=np.uint8) % 2,
        "p2_button_a": (np.arange(length, dtype=np.uint8) + 1) % 2,
        "p1_percent": np.zeros(length, dtype=np.float32),
        "p2_percent": np.zeros(length, dtype=np.float32),
        "p1_stock": np.full(length, 4, dtype=np.int32),
        "p2_stock": np.full(length, 4, dtype=np.int32),
    }


def test_context_window_ends_at_requested_frame_and_pads_only_the_past() -> None:
    sample = _sample()

    first = value_meter._context_window(sample, ego_prefix="p1", end_index=0, context_length=4)
    later = value_meter._context_window(sample, ego_prefix="p2", end_index=3, context_length=4)

    assert first["ctx_pad"] == 3
    np.testing.assert_array_equal(first["ego_position_x"], [0, 0, 0, 10])
    np.testing.assert_array_equal(first["opp_position_x"], [0, 0, 0, 20])
    np.testing.assert_array_equal(first["ego_button_a"], [0, 0, 0, 0])
    np.testing.assert_array_equal(later["ego_position_x"], [20, 21, 22, 23])
    np.testing.assert_array_equal(later["opp_position_x"], [10, 11, 12, 13])
    np.testing.assert_array_equal(later["ego_button_a"], [1, 0, 1, 0])
    np.testing.assert_array_equal(later["ego_player_id"], [0, 0, 0, 0])


def test_context_window_does_not_read_a_future_frame() -> None:
    sample = _sample()
    baseline = value_meter._context_window(sample, ego_prefix="p1", end_index=2, context_length=4)
    sample["p1_position_x"][3:] = 999

    actual = value_meter._context_window(sample, ego_prefix="p1", end_index=2, context_length=4)

    np.testing.assert_array_equal(actual["ego_position_x"], baseline["ego_position_x"])


def test_validate_frames_rejects_a_gap() -> None:
    sample = _sample()
    sample["frame"][3] += 1

    with pytest.raises(ValueError, match=r"not contiguous at -121 -> -119"):
        value_meter._validate_frames(sample)


def test_pair_values_keeps_frame_major_perspective_order() -> None:
    p1, p2 = value_meter._pair_values(np.array([1.0, -1.0, 2.0, -2.0]), 2)

    assert p1 == (1.0, 2.0)
    assert p2 == (-1.0, -2.0)


def test_pair_values_rejects_nonfinite_output() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        value_meter._pair_values(np.array([1.0, np.nan]), 1)


def test_retrospective_trace_moves_a_future_reward_back_to_prior_actions() -> None:
    series = value_meter.ValueSeries((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)

    trace = value_meter.retrospective_trace(
        series,
        np.array([0.0, 0.0, 4.0]),
        gamma=0.5,
        half_life_frames=1.0,
    )

    assert trace.value == pytest.approx((2.0, 4.0, 0.0))
    assert trace.reward == (0.0, 0.0, 4.0)
    assert trace.gae_lambda == pytest.approx(1.0)
    assert trace.decay == pytest.approx(0.5)


def test_retrospective_trace_rejects_half_life_that_requires_lambda_above_one() -> None:
    series = value_meter.ValueSeries((0.0, 0.0), (0.0, 0.0), 1.0)

    with pytest.raises(ValueError, match="gamma-limited maximum"):
        value_meter.retrospective_trace(
            series,
            np.zeros(2),
            gamma=0.5,
            half_life_frames=2.0,
        )


def test_resolve_local_replay_records_identity(tmp_path: Path) -> None:
    replay = tmp_path / "Game.slp"
    replay.write_bytes(b"replay bytes")

    resolved = value_meter.resolve_replay(str(replay))

    assert resolved.path == replay.resolve()
    assert resolved.size == len(b"replay bytes")
    assert resolved.sha256 == hashlib.sha256(b"replay bytes").hexdigest()
    assert resolved.etag is None


def test_resolve_r2_replay_downloads_and_reuses_a_verified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"remote replay"
    client = MagicMock()
    client.head_object.return_value = {"ETag": '"etag"', "ContentLength": len(payload)}
    client.get_object.return_value = {"ETag": '"etag"', "Body": io.BytesIO(payload)}
    monkeypatch.setattr(value_meter.r2, "client", lambda: client)

    first = value_meter.resolve_replay("r2://hal/runs/Game.slp", cache_root=tmp_path)
    client.get_object.reset_mock()
    second = value_meter.resolve_replay("r2://hal/runs/Game.slp", cache_root=tmp_path)

    assert first == second
    assert first.path.read_bytes() == payload
    assert first.sha256 == hashlib.sha256(payload).hexdigest()
    metadata_path = first.path.with_name(f"{first.path.name}.metadata.json")
    assert json.loads(metadata_path.read_text()) == {
        "etag": '"etag"',
        "sha256": first.sha256,
        "size": len(payload),
        "uri": "r2://hal/runs/Game.slp",
    }
    client.get_object.assert_not_called()


def test_resolve_r2_replay_rejects_a_non_replay_object() -> None:
    with pytest.raises(ValueError, match="must name one .slp object"):
        value_meter.resolve_replay("r2://hal/runs/manifest.json")


@dataclass(frozen=True)
class _Calibration:
    gamma: float = 0.99855
    stock_value: float = 120.0
    damage_shaping: float = 1.0
    win_reward: float = 50.0


@dataclass(frozen=True)
class _Config:
    awr: _Calibration = _Calibration()


def test_build_sidecar_records_model_and_replay_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    replay_path = tmp_path / "Game.slp"
    replay_path.write_bytes(b"replay")
    replay = value_meter.ResolvedReplay(
        replay_path,
        "r2://hal/Game.slp",
        hashlib.sha256(b"replay").hexdigest(),
        len(b"replay"),
        '"etag"',
    )
    monkeypatch.setattr(value_meter, "_git_sha", lambda: "a" * 40)

    sidecar = value_meter.build_sidecar(
        checkpoint_source="r2://hal/checkpoint.pt",
        checkpoint_path=checkpoint,
        replay=replay,
        sample=_sample(2),
        series=value_meter.ValueSeries((12.0, 24.0), (-12.0, -24.0), 0.5),
        cfg=_Config(),
        state={"cfg": {"experiment_id": value_meter._EXPERIMENT_ID}, "step": 16_383},
        credit_half_life_frames=45.0,
    )

    assert sidecar["schema_version"] == 2
    assert sidecar["checkpoint"]["step"] == 16_383  # type: ignore[index]
    assert sidecar["replay"]["first_frame"] == -123  # type: ignore[index]
    assert sidecar["value"]["projection"] == "(p1-p2)/2"  # type: ignore[index]
    assert sidecar["credit"]["method"] == "retrospective_v_plus_gae"  # type: ignore[index]
    assert sidecar["series"]["reward_p1"] == (0.0, 0.0)  # type: ignore[index]


def test_build_sidecar_rejects_wrong_experiment(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    replay_path = tmp_path / "Game.slp"
    replay_path.write_bytes(b"replay")
    replay = value_meter.ResolvedReplay(replay_path, str(replay_path), "b" * 64, 6, None)

    with pytest.raises(ValueError, match="checkpoint is not"):
        value_meter.build_sidecar(
            checkpoint_source=str(checkpoint),
            checkpoint_path=checkpoint,
            replay=replay,
            sample=_sample(1),
            series=value_meter.ValueSeries((0.0,), (0.0,), 1.0),
            cfg=_Config(),
            state={"cfg": {"experiment_id": "other"}, "step": 1},
            credit_half_life_frames=45.0,
        )
