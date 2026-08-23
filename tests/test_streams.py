from pathlib import Path

import pytest

from hal import streams
from hal.data.feature_stats import FeatureStatsSufficient
from hal.data.feature_stats import dump_sufficient_stats
from hal.training.ego_stats import load_consolidated_mixture_stats
from hal.training.ego_stats import load_consolidated_stats


def test_policy_world_manifest_has_the_verified_natural_mix() -> None:
    names = {source.name for source in streams.POLICY_WORLD_V7_SOURCES}

    assert len(streams.POLICY_WORLD_V7_SOURCES) == len(names) == 35
    assert names == set(streams.POLICY_WORLD_V7_TRAIN_REPLAYS)
    assert names == set(streams.POLICY_WORLD_V7_TRAIN_FRAMES)
    assert sum(streams.POLICY_WORLD_V7_TRAIN_REPLAYS.values()) == 1_213_707
    assert sum(streams.POLICY_WORLD_V7_TRAIN_FRAMES.values()) == 12_391_036_805


def test_stats_loader_pulls_only_selected_registered_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = streams.RANKED_ANONYMIZED_1_POLICY_V7
    calls: list[streams.StreamSource] = []
    monkeypatch.setattr(streams, "REPO_DIR", str(tmp_path))

    def fake_pull_stats(src: streams.StreamSource) -> Path:
        calls.append(src)
        destination = src.local_root / "stats.json"
        destination.parent.mkdir(parents=True)
        sufficient = {
            "p1_x": FeatureStatsSufficient(count=2, mean=1.0, m2=2.0, min=0.0, max=2.0),
        }
        dump_sufficient_stats(destination, sufficient, split="train", mds_schema_version=7)
        return destination

    monkeypatch.setattr(streams, "pull_stats", fake_pull_stats)

    result = load_consolidated_stats(tmp_path / selected.local / "stats.json")

    assert result["x"].mean == 1.0
    assert calls == [selected]
    assert streams.remote_for_local(selected.local) == selected.remote
    assert streams.remote_for_local(tmp_path / selected.local) == selected.remote


def test_ensure_stats_leaves_existing_and_unregistered_paths_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "local" / "stats.json"
    existing.parent.mkdir()
    existing.write_text("{}")

    def fail_pull_stats(_src: streams.StreamSource) -> Path:
        raise AssertionError("unexpected download")

    monkeypatch.setattr(streams, "pull_stats", fail_pull_stats)

    assert streams.ensure_stats(existing) == existing
    assert streams.ensure_stats(tmp_path / "missing" / "stats.json") == tmp_path / "missing" / "stats.json"


def test_consolidated_mixture_stats_use_source_proportions(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    dump_sufficient_stats(
        first,
        {
            "p1_x": FeatureStatsSufficient(count=2, mean=0.0, m2=2.0, min=-1.0, max=1.0),
            "p2_x": FeatureStatsSufficient(count=2, mean=0.0, m2=2.0, min=-1.0, max=1.0),
        },
        split="train",
        mds_schema_version=7,
    )
    dump_sufficient_stats(
        second,
        {
            "p1_x": FeatureStatsSufficient(count=2, mean=4.0, m2=8.0, min=2.0, max=6.0),
            "p2_x": FeatureStatsSufficient(count=2, mean=4.0, m2=8.0, min=2.0, max=6.0),
        },
        split="train",
        mds_schema_version=7,
    )

    stats = load_consolidated_mixture_stats([first, second], [0.25, 0.75], expected_mds_schema_version=7)["x"]

    assert stats.mean == pytest.approx(3.0)
    assert stats.std == pytest.approx(2.5)
    assert stats.min == -1.0
    assert stats.max == 6.0


def test_consolidated_mixture_stats_check_every_schema(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    sufficient = {"p1_x": FeatureStatsSufficient(count=1, mean=0.0, m2=0.0, min=0.0, max=0.0)}
    dump_sufficient_stats(first, sufficient, split="train", mds_schema_version=7)
    dump_sufficient_stats(second, sufficient, split="train", mds_schema_version=6)

    with pytest.raises(ValueError, match="mds_schema_version"):
        load_consolidated_mixture_stats([first, second], [1.0, 1.0], expected_mds_schema_version=7)
