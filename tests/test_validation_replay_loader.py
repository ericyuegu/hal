"""059 validation window identity and Mosaic configuration."""

from collections.abc import Iterator
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
import pytest
from streaming import StreamingDataset

import hal.training.validation_replay_loader as validation
from hal.data.schema import SCHEMA_VERSION
from hal.representation.features import ITEM_PLAYER_COLUMNS
from hal.representation.features import FeatureProjection
from hal.streams import StreamSource
from hal.training.replay_windows import Window


def _decode(row: Mapping[str, object]) -> dict[str, np.ndarray | int]:
    frames = cast(int, row["num_frames"])
    return {
        "schema_version": SCHEMA_VERSION,
        "frame": np.arange(frames, dtype=np.int32),
        "p1_value": np.arange(frames, dtype=np.float32),
        "p2_value": np.arange(frames, dtype=np.float32) + 100,
    }


def _labels(row: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {"value_label": np.arange(cast(int, row["num_frames"]), dtype=np.int64) + 200}


def _identity_batch_transform(_windows: list[Window], batch: object) -> object:
    return batch


def test_validation_windows_match_control_replay_start_ego_and_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "decode_policy_world_replay", _decode)
    rows = [
        {"replay_id": "replay-2", "num_frames": 12},
        {"replay_id": "replay-4", "num_frames": 30},
        {"replay_id": "replay-7", "num_frames": 37},
    ]
    projection = FeatureProjection(frozenset({"frame", "ego_value", "opp_value", "value_label"}))
    windows = validation._ValidationWindows(
        cast(StreamingDataset, rows),
        context_length=4,
        chunk_length=3,
        seed=0,
        schema_version=SCHEMA_VERSION,
        projection=projection,
        replay_labels=_labels,
    )

    output = list(windows)

    assert len(output) == 3
    assert windows.identities == [
        validation.ValidationWindowIdentity(
            "replay-2", 5, "p1", "23fe98f19a9460456851b271dc651b708de39be00e111ca19590e34142d327f3"
        ),
        validation.ValidationWindowIdentity(
            "replay-4", 14, "p1", "19c4af1bddf2397122d11450fd719994d4a7c7ce0ddda6e829e8b012db0d5fe1"
        ),
        validation.ValidationWindowIdentity(
            "replay-7", 3, "p2", "b66147d68fb964fb43f486ef8b7ad096956f20a35691027143b4488e0253020a"
        ),
    ]
    assert all(int(window["ctx_pad"]) == 0 for window in output)
    assert all(len(window["frame"]) == 7 for window in output)


def test_validation_rejects_short_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validation, "decode_policy_world_replay", _decode)
    windows = validation._ValidationWindows(
        cast(StreamingDataset, [{"replay_id": "too-short", "num_frames": 6}]),
        context_length=4,
        chunk_length=3,
        seed=0,
        schema_version=SCHEMA_VERSION,
        projection=FeatureProjection(frozenset({"frame"})),
        replay_labels=_labels,
    )

    with pytest.raises(ValueError, match="full-context window requires at least 7 frames"):
        list(windows)


def test_validation_loader_pins_059_mosaic_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, Any]] = []

    def fake_stream(**kwargs: object) -> object:
        created.append(dict(kwargs))
        return object()

    class FakeDataset:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __iter__(self) -> Iterator[dict[str, object]]:
            return iter(())

    monkeypatch.setattr(validation, "patch_streaming", lambda: None)
    monkeypatch.setattr(validation, "Stream", fake_stream)
    monkeypatch.setattr(validation, "StreamingDataset", FakeDataset)
    sources = (
        StreamSource("first", "s3://bucket/first", Path("first")),
        StreamSource("second", "s3://bucket/second", Path("second")),
    )

    loader = validation.make_validation_replay_loader(
        sources=sources,
        stats={},
        context_length=256,
        chunk_length=28,
        batch_size=128,
        seed=0,
        cache_limit="1792gb",
        schema_version=SCHEMA_VERSION,
        extra=ITEM_PLAYER_COLUMNS,
        projection=FeatureProjection(frozenset({"frame"})),
        replay_labels=_labels,
        batch_transform=_identity_batch_transform,
    )

    assert [item["remote"] for item in created] == [source.remote for source in sources]
    assert all(item["split"] == "val" and item["repeat"] == 1 for item in created)
    dataset = cast(validation._ValidationWindows, loader.dataset)
    settings = dict(cast(FakeDataset, dataset.mds).kwargs)
    assert len(cast(list[object], settings.pop("streams"))) == 2
    assert settings == {
        "cache_limit": "1792gb",
        "predownload": 1024,
        "batch_size": 128,
        "shuffle": True,
        "shuffle_algo": "py1e",
        "shuffle_block_size": 8192,
        "shuffle_seed": 0,
    }
    assert loader.num_workers == 0
    assert dataset.context_length == 256
    assert dataset.chunk_length == 28
