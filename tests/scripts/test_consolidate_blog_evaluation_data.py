from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

_PATH = Path(__file__).parents[2] / "scripts" / "consolidate_blog_evaluation_data.py"
_SPEC = importlib.util.spec_from_file_location("consolidate_blog_evaluation_data", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

aggregate_boots = _MODULE.aggregate_boots
difference_statistics = _MODULE.difference_statistics
evaluation_statistics = _MODULE.evaluation_statistics


def _row(
    *,
    boot: int,
    ordinal: int,
    frames: int,
    stocks_taken: int,
    stocks_lost: int,
    damage_dealt: float,
    damage_taken: float,
) -> dict[str, int | float]:
    return {
        "active_frames": frames,
        "boot_index": boot,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
        "ego_character": 1,
        "match_ordinal": ordinal,
        "opp_character": 2,
        "stage": 24,
        "stocks_lost": stocks_lost,
        "stocks_taken": stocks_taken,
        "total_frames": frames + 123,
    }


def test_boot_aggregation_is_the_independent_unit() -> None:
    rows = [
        _row(
            boot=0,
            ordinal=0,
            frames=1800,
            stocks_taken=2,
            stocks_lost=1,
            damage_dealt=100,
            damage_taken=40,
        ),
        _row(
            boot=0,
            ordinal=1,
            frames=1800,
            stocks_taken=1,
            stocks_lost=0,
            damage_dealt=50,
            damage_taken=20,
        ),
        _row(
            boot=1,
            ordinal=0,
            frames=3600,
            stocks_taken=1,
            stocks_lost=2,
            damage_dealt=20,
            damage_taken=50,
        ),
    ]

    boots = aggregate_boots(rows)
    statistics = evaluation_statistics(boots)

    assert len(boots) == 2
    assert boots[0]["matches"] == 2
    assert boots[0]["stocks_taken"] == 3
    assert boots[0]["active_frames"] == 3600
    assert math.isclose(statistics["net_stocks_per_min"]["pooled_mean"], 0.5)
    assert statistics["net_stocks_per_min"]["boot_rate_sd"] > 0


def test_difference_uses_matching_boots_as_blocks() -> None:
    control = {
        "boots": aggregate_boots(
            [
                _row(
                    boot=index,
                    ordinal=0,
                    frames=3600,
                    stocks_taken=1,
                    stocks_lost=1,
                    damage_dealt=50,
                    damage_taken=50,
                )
                for index in range(4)
            ]
        )
    }
    treatment = {
        "boots": aggregate_boots(
            [
                _row(
                    boot=index,
                    ordinal=0,
                    frames=3600,
                    stocks_taken=2,
                    stocks_lost=1,
                    damage_dealt=60,
                    damage_taken=50,
                )
                for index in range(4)
            ]
        )
    }

    stock = difference_statistics(treatment, control, "stock")
    damage = difference_statistics(treatment, control, "damage")

    assert stock["method"] == "matchup-blocked bootstrap of shared boots"
    assert stock["matchup_pairing_fraction"] == 1.0
    assert stock["point_delta"] == 1.0
    assert stock["ci95"] == [1.0, 1.0]
    assert damage["point_delta"] == 10.0


def test_committed_evidence_references_unique_evaluations() -> None:
    data_path = Path(__file__).parents[2] / "docs" / "blog_workbench" / "evidence.json"
    document = json.loads(data_path.read_text())
    raw = document["raw_evaluation_data"]
    evaluations = raw["evaluations"]

    assert document["schema_version"] == 2
    assert raw["deduplication"]["unique_evaluations"] == len(evaluations)
    assert raw["deduplication"]["duplicate_source_files_collapsed"] > 0
    for evaluation_id, evaluation in evaluations.items():
        assert evaluation["evaluation_id"] == evaluation_id
        assert evaluation["statistics"]["independent_unit"] == "Dolphin boot"
    for study in document["studies"]:
        for link in study.get("evaluation_links", []):
            selected = link.get("selected_evaluation_id")
            assert selected is None or selected in evaluations
