import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from hal import streams
from hal.data.policy_schema import PACKED_STATE_SUFFIXES
from hal.data.policy_schema import pack_player_state
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.training import returns as returns_lib
from hal.training.player_identity import ReplayPlayerLookup


def _load_experiment() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "experiments" / "051_muon_parameterization.py"
    spec = importlib.util.spec_from_file_location("test_o51_data_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load_experiment()
CORPUS_SHA256 = exp.CORPUS_SHA256
D0 = exp.D0
EXCLUDED_SOURCE_ROWS = exp.EXCLUDED_SOURCE_ROWS
O51_RETURN_SUFFIX = exp.O51_RETURN_SUFFIX
OFFICIAL_CORPUS_TARGETS = exp.OFFICIAL_CORPUS_TARGETS
OFFICIAL_RAW_REPLAYS = exp.OFFICIAL_RAW_REPLAYS
OFFICIAL_TIER_REPLAYS = exp.OFFICIAL_TIER_REPLAYS
OFFICIAL_TIER_TARGETS = exp.OFFICIAL_TIER_TARGETS
OFFICIAL_UNIQUE_REPLAYS = exp.OFFICIAL_UNIQUE_REPLAYS
SOURCE_MANIFEST_SHA256 = exp.SOURCE_MANIFEST_SHA256
TIER_SHA256 = exp.TIER_SHA256
ParameterizationReplayLabels = exp.ParameterizationReplayLabels
corpus_selection = exp.corpus_selection


def test_official_direct_data_accounting_is_pinned() -> None:
    assert D0 == 2**30
    assert (OFFICIAL_RAW_REPLAYS, OFFICIAL_UNIQUE_REPLAYS) == (
        1_300_640,
        1_300_638,
    )
    assert OFFICIAL_CORPUS_TARGETS == 26_582_742_076
    assert OFFICIAL_TIER_REPLAYS == {
        1: 162_598,
        2: 325_176,
        4: 650_331,
        8: 1_300_638,
    }
    assert OFFICIAL_TIER_TARGETS == {
        1: 3_353_805_100,
        2: 6_686_081_812,
        4: 13_291_247_716,
        8: 26_582_742_076,
    }


def test_tiers_are_direct_source_prefixes_with_pinned_hashes() -> None:
    selection = corpus_selection()
    source_names = [source.name for source in streams.POLICY_WORLD_V7_SOURCES]

    assert selection.corpus_hash == CORPUS_SHA256
    assert selection.source_manifest_sha256 == {name: SOURCE_MANIFEST_SHA256[name] for name in source_names}
    assert {scale: tier.sha256 for scale, tier in selection.tiers.items()} == TIER_SHA256
    assert {scale: tier.unique_replays for scale, tier in selection.tiers.items()} == OFFICIAL_TIER_REPLAYS
    assert {scale: tier.potential_targets for scale, tier in selection.tiers.items()} == OFFICIAL_TIER_TARGETS
    for tier in selection.tiers.values():
        assert [source.source for source in tier.sources] == source_names
        assert tier.frames == tier.potential_targets // 2 + tier.unique_replays


def test_only_full_tier_uses_the_two_row_exclusion_sidecar() -> None:
    selection = corpus_selection()
    source = "professional-monotheon-policy-world-v7"
    selected = {
        scale: next(view for view in tier.sources if view.source == source) for scale, tier in selection.tiers.items()
    }

    assert {source: (14_160, 14_163)} == EXCLUDED_SOURCE_ROWS
    assert {scale: view.stop for scale, view in selected.items()} == {
        1: 2_042,
        2: 4_083,
        4: 8_166,
        8: 16_333,
    }
    assert all(not selected[scale].excluded_rows for scale in (1, 2, 4))
    assert selected[8].excluded_rows == EXCLUDED_SOURCE_ROWS[source]


def _compact_replay(frames: int, *, terminal: bool) -> dict[str, object]:
    compact: dict[str, object] = {}
    for name, encoding in POLICY_WORLD_MDS_COLUMNS.items():
        if encoding == "str":
            compact[name] = "replay-1"
        elif encoding == "int":
            compact[name] = 0
        else:
            dtype = np.dtype(encoding.removeprefix("ndarray:"))
            compact[name] = np.zeros(frames, dtype=dtype)
    compact.update(
        policy_world_schema_version=POLICY_WORLD_SCHEMA_VERSION,
        source_schema_version=7,
        replay_id="replay-1",
        num_frames=frames,
    )
    for port in ("p1", "p2"):
        values = {name: np.zeros(frames, dtype=np.int32) for name in PACKED_STATE_SUFFIXES}
        values["stock"].fill(4)
        values["direction"] = np.ones(frames, dtype=np.float32)
        if terminal and port == "p2":
            values["stock"][-1] = 0
        compact[f"{port}_state"] = pack_player_state(values)
        compact[f"{port}_percent"] = np.arange(frames, dtype=np.float32)
    return compact


def test_direct_labels_compute_returns_and_ids_from_existing_row() -> None:
    compact = _compact_replay(12, terminal=True)
    expected = returns_lib.compact_policy_returns(
        compact,
        gamma=0.9,
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
        suffix=O51_RETURN_SUFFIX,
    )
    labels = ParameterizationReplayLabels(
        player_lookup=ReplayPlayerLookup({"replay-1": (7, 11)}),
        gamma=0.9,
        damage_shaping=1.0,
        win_reward=50.0,
        stock_value=120.0,
    )(compact)

    for name, values in expected.items():
        np.testing.assert_array_equal(labels[name], values)
    np.testing.assert_array_equal(labels["p1_player_id"], np.asarray(7, dtype=np.int32))
    np.testing.assert_array_equal(labels["p2_player_id"], np.asarray(11, dtype=np.int32))
