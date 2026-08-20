from pathlib import Path

import pytest

from hal import streams
from hal.data.feature_stats import FeatureStatsSufficient
from hal.data.feature_stats import dump_sufficient_stats
from hal.training.ego_stats import load_consolidated_stats


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
